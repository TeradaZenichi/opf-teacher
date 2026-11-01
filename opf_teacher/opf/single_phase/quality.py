"""Check scaled feasibility and physical device consistency after a solve."""
import math

import numpy as np
import pyomo.environ as pyo


def analyze_quality(model, case, tolerance=1e-6):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Quality tolerance must be finite and positive")
    scale = model.scaling_factor
    violations = {}
    for component in model.component_data_objects((pyo.Constraint, pyo.Var), active=True):
        value = pyo.value(component.body if component.ctype is pyo.Constraint else component, exception=False)
        if value is None or not math.isfinite(value):
            violations[component.name] = float("inf")
            continue
        lower = component.lower if component.ctype is pyo.Constraint else component.lb
        upper = component.upper if component.ctype is pyo.Constraint else component.ub
        low = 0.0 if lower is None else max(pyo.value(lower) - value, 0.0)
        high = 0.0 if upper is None else max(value - pyo.value(upper), 0.0)
        factor = scale.get(component, scale.get(component.parent_component(), 1.0))
        violations[component.name] = max(low, high) * factor
    loss_deficits, energy_errors, execution_errors, availability_errors, simultaneous = [], [], [], [], []
    consistency = []
    for battery in case.bess:
        result = battery.result
        previous = result.soc_kwh.shift(fill_value=battery.soc_init_frac * battery.e_cap_kwh)
        physical_loss = battery.q_loss_rated_kw * (result.q_kvar / battery.s_max_kva) ** 2
        gain = battery.eta_charge * result.charge_kw - result.discharge_kw / battery.eta_discharge - result.inverter_loss_kw
        net = result.charge_kw - result.discharge_kw
        physical_gain = battery.eta_charge * net.clip(lower=0) + net.clip(upper=0) / battery.eta_discharge - physical_loss
        error = (result.soc_kwh - previous - physical_gain * case.dt_h).abs().max(skipna=False)
        energy_errors.append((result.soc_kwh - previous - gain * case.dt_h).abs().max(skipna=False) * 1000)
        execution_errors.append(error * 1000)
        consistency.append(error / battery.e_cap_kwh)
        loss_deficits.append((physical_loss - result.inverter_loss_kw).clip(lower=0).max(skipna=False) * 1000)
        simultaneous.append(np.minimum(result.charge_kw, result.discharge_kw).max(skipna=False))
    for pv in case.pv:
        result = pv.result
        physical_loss = pv.q_loss_rated_kw * (result.q_kvar / pv.s_max_kva) ** 2
        loss_deficits.append((physical_loss - result.inverter_loss_kw).clip(lower=0).max(skipna=False) * 1000)
        solar_loss = physical_loss.where(result.avail_kw > 0, 0.0)
        error = (result.gen_kw + solar_loss - result.avail_kw).clip(lower=0).max(skipna=False)
        availability_errors.append(error * 1000)
        consistency.append(error / (pv.p_max_kw or 1.0))
    maximum = max(violations.values(), default=0.0)
    feasible = maximum <= tolerance
    consistent = all(value <= tolerance for value in consistency)
    return {
        "tolerance": tolerance, "max_scaled_violation": maximum,
        "worst_constraint": max(violations, key=violations.get, default=None),
        "loss_underestimate_max_w": max(loss_deficits, default=0.0),
        "energy_balance_error_max_wh": max(energy_errors, default=0.0),
        "energy_execution_error_max_wh": max(execution_errors, default=0.0),
        "pv_availability_violation_max_w": max(availability_errors, default=0.0),
        "simultaneous_charge_discharge_max_kw": max(simultaneous, default=0.0),
        "feasible": feasible, "device_consistent": consistent,
        "valid_for_imitation": feasible and consistent and case.summary.socp_relaxation_tight,
    }
