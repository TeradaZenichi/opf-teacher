"""Single-phase active-power LinDistFlow model, without reactive power or losses."""
from __future__ import annotations

import math

import pandas as pd
import pyomo.environ as pyo

from opf.components import (BessResult, BranchResult, BusResult, Case, GridResult, PvResult, Summary)


DEFAULT_TIE_BREAK_PENALTY = 1e-6


def build_active_power_model(case: Case, tie_break_penalty: float = DEFAULT_TIE_BREAK_PENALTY) -> pyo.ConcreteModel:
    if tie_break_penalty < 0.0:
        raise ValueError("tie_break_penalty cannot be negative")

    m = pyo.ConcreteModel(name=f"{case.name}-active-power")
    base = case.base
    dt = case.dt_h
    periods = list(case.periods)
    previous = {periods[i]: periods[i - 1] for i in range(1, len(periods))}
    non_root = [bus for bus in case.buses if bus != case.root]

    branch_at = {bus: case.branches[case.parent_branch[bus]] for bus in non_root}
    resistance = {bus: branch_at[bus].impedance_pu(base, case.buses[branch_at[bus].from_bus])[0] for bus in non_root}
    tap = {bus: branch_at[bus].tap_ratio for bus in non_root}
    branch_limit = {bus: base.pu_power(branch_at[bus].s_max_kva) for bus in non_root}
    load = {bus: base.pu_power(case.buses[bus].p_load_kw.to_numpy()) for bus in case.buses}
    price = case.price.to_numpy()

    bess = {device.id: device for device in case.bess}
    pv = {device.id: device for device in case.pv}
    bess_at = {bus: [device for device in case.bess if device.bus == bus] for bus in case.buses}
    pv_at = {bus: [device for device in case.pv if device.bus == bus] for bus in case.buses}
    pv_available = {device.id: base.pu_power(device.avail_kw.to_numpy()) for device in case.pv}

    m.T = pyo.Set(initialize=periods, ordered=True)
    m.B = pyo.Set(initialize=list(case.buses))
    m.J = pyo.Set(initialize=non_root, ordered=True)
    m.S = pyo.Set(initialize=list(bess))
    m.G = pyo.Set(initialize=list(pv))

    def voltage_bounds(_m, bus, _t):
        device_bus = case.buses[bus]
        if bus == case.root:
            fixed = case.grid.v_ref_pu ** 2
            return fixed, fixed
        return device_bus.v_min_pu ** 2, device_bus.v_max_pu ** 2

    m.v = pyo.Var(m.B, m.T, bounds=voltage_bounds)
    m.P = pyo.Var(m.J, m.T, bounds=lambda _m, bus, _t: (-branch_limit[bus], branch_limit[bus]))
    m.pimp = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, base.pu_power(case.grid.p_import_max_kw)))
    m.pexp = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, base.pu_power(case.grid.p_export_max_kw)))

    m.pch = pyo.Var(
        m.S, m.T, domain=pyo.NonNegativeReals,
        bounds=lambda _m, device_id, _t: (
            0.0,
            base.pu_power(min(bess[device_id].p_charge_max_kw, bess[device_id].s_max_kva)),
        ),
    )
    m.pdis = pyo.Var(
        m.S, m.T, domain=pyo.NonNegativeReals,
        bounds=lambda _m, device_id, _t: (
            0.0,
            base.pu_power(min(bess[device_id].p_discharge_max_kw, bess[device_id].s_max_kva)),
        ),
    )
    m.soc = pyo.Var(
        m.S, m.T,
        bounds=lambda _m, device_id, _t: (
            bess[device_id].soc_min_frac * base.pu_energy(bess[device_id].e_cap_kwh),
            bess[device_id].soc_max_frac * base.pu_energy(bess[device_id].e_cap_kwh),
        ),
    )

    def pv_bounds(_m, device_id, t):
        device = pv[device_id]
        upper = min(pv_available[device_id][t], base.pu_power(device.p_max_kw), base.pu_power(device.s_max_kva))
        return (0.0, upper) if device.curtailable else (upper, upper)

    m.ppv = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals, bounds=pv_bounds)

    def injection(_m, bus, t):
        value = -load[bus][t]
        value += sum(_m.pdis[device.id, t] - _m.pch[device.id, t] for device in bess_at[bus])
        value += sum(_m.ppv[device.id, t] for device in pv_at[bus])
        if bus == case.root:
            value += _m.pimp[t] - _m.pexp[t]
        return value

    def active_balance(_m, bus, t):
        outgoing = sum(_m.P[child, t] for child in case.children.get(bus, []))
        incoming = _m.P[bus, t] if bus != case.root else 0.0
        return incoming + injection(_m, bus, t) == outgoing

    m.active_balance = pyo.Constraint(m.B, m.T, rule=active_balance)

    def voltage_drop(_m, bus, t):
        parent = branch_at[bus].from_bus
        return _m.v[bus, t] == tap[bus] ** 2 * (_m.v[parent, t] - 2.0 * resistance[bus] * _m.P[bus, t])

    m.voltage_drop = pyo.Constraint(m.J, m.T, rule=voltage_drop)

    def soc_balance(_m, device_id, t):
        device = bess[device_id]
        delta = (device.eta_charge * _m.pch[device_id, t] - _m.pdis[device_id, t] / device.eta_discharge) * dt
        initial = (device.soc_init_frac * base.pu_energy(device.e_cap_kwh))
        prior = initial if t == periods[0] else _m.soc[device_id, previous[t]]
        return _m.soc[device_id, t] == prior + delta

    m.soc_balance = pyo.Constraint(m.S, m.T, rule=soc_balance)

    def terminal_soc(_m, device_id):
        device = bess[device_id]
        if not device.cyclic_soc and device.soc_terminal_frac is None:
            return pyo.Constraint.Skip
        target = (device.soc_init_frac if device.soc_terminal_frac is None else device.soc_terminal_frac)
        return _m.soc[device_id, periods[-1]] == (target * base.pu_energy(device.e_cap_kwh))

    m.soc_terminal = pyo.Constraint(m.S, rule=terminal_soc)

    feed_in_ratio = case.grid.feed_in_ratio
    m.energy_cost = pyo.Expression(
        expr=base.s_base_kva * dt * sum(price[t] * (m.pimp[t] - feed_in_ratio * m.pexp[t]) for t in periods)
    )
    m.tie_break_cost = pyo.Expression(
        expr=tie_break_penalty * base.s_base_kva * dt * (
            sum(m.pimp[t] + m.pexp[t] for t in periods)
            + sum(m.pch[device_id, t] + m.pdis[device_id, t] for device_id in m.S for t in periods)
        )
    )
    m.cost = pyo.Objective(expr=m.energy_cost + m.tie_break_cost, sense=pyo.minimize)
    return m


def solve_active_power_opf(
    case: Case,
    solver: str = "appsi_highs",
    tie_break_penalty: float = DEFAULT_TIE_BREAK_PENALTY,
) -> Case:
    model = build_active_power_model(case, tie_break_penalty=tie_break_penalty)
    optimizer = pyo.SolverFactory(solver)
    if optimizer is None or not optimizer.available(exception_flag=False):
        raise RuntimeError(f"Solver {solver!r} unavailable")
    result = optimizer.solve(model)
    termination = str(result.solver.termination_condition)
    if termination.lower() not in {"optimal", "locallyoptimal", "feasible"}:
        raise RuntimeError(f"Active-power OPF did not solve: {termination}")
    return _attach_active_results(model, case, termination)


def _attach_active_results(model, case: Case, status: str) -> Case:
    base = case.base
    index = case.index
    periods = list(case.periods)

    def value(expression) -> float:
        return float(pyo.value(expression))

    def series(values) -> pd.Series:
        return pd.Series(values, index=index, dtype=float)

    zeros = series([0.0] * len(periods))
    imported = series([base.to_kw(value(model.pimp[t])) for t in periods])
    exported = series([base.to_kw(value(model.pexp[t])) for t in periods])
    grid_cost = (imported - case.grid.feed_in_ratio * exported) * case.price * case.dt_h
    case.grid.result = GridResult(imported, exported, zeros.copy(), grid_cost)

    for bus_id, bus in case.buses.items():
        bus.result = BusResult(series([math.sqrt(max(value(model.v[bus_id, t]), 0.0)) for t in periods]))

    for bus_id in (bus for bus in case.buses if bus != case.root):
        branch = case.branches[case.parent_branch[bus_id]]
        branch.result = BranchResult(
            p_kw=series([base.to_kw(value(model.P[bus_id, t])) for t in periods]),
            q_kvar=zeros.copy(),
            loss_kw=zeros.copy(),
            socp_gap_pu2=None,
            socp_gap_normalized=None,
        )

    for device in case.bess:
        charge = series([base.to_kw(value(model.pch[device.id, t])) for t in periods])
        discharge = series([base.to_kw(value(model.pdis[device.id, t])) for t in periods])
        soc = series([base.to_kwh(value(model.soc[device.id, t])) for t in periods])
        device.result = BessResult(
            charge_kw=charge,
            discharge_kw=discharge,
            p_net_kw=charge - discharge,
            soc_kwh=soc,
            soc_frac=soc / device.e_cap_kwh,
            q_kvar=zeros.copy(),
            inverter_loss_kw=zeros.copy(),
        )

    for device in case.pv:
        generation = series([base.to_kw(value(model.ppv[device.id, t])) for t in periods])
        device.result = PvResult(
            avail_kw=device.avail_kw.copy(),
            gen_kw=generation,
            curtail_kw=device.avail_kw - generation,
            q_kvar=zeros.copy(),
            inverter_loss_kw=zeros.copy(),
            p_net_kw=generation.copy(),
            grid_consumption_kw=zeros.copy(),
        )

    voltages = [float(bus.result.v_pu.min()) for bus in case.buses.values()]
    voltage_maxima = [float(bus.result.v_pu.max()) for bus in case.buses.values()]
    case.summary = Summary(
        status=status,
        objective_cost=value(model.cost),
        energy_cost=float(grid_cost.sum()),
        energy_import_kwh=float(imported.sum()) * case.dt_h,
        energy_export_kwh=float(exported.sum()) * case.dt_h,
        pv_generated_kwh=sum(float(device.result.gen_kw.sum()) for device in case.pv) * case.dt_h,
        pv_curtailed_kwh=sum(float(device.result.curtail_kw.sum()) for device in case.pv) * case.dt_h,
        losses_kwh=0.0,
        v_min_pu=min(voltages),
        v_max_pu=max(voltage_maxima),
        socp_gap_max_pu2=float("nan"),
        socp_gap_max_normalized=float("nan"),
        socp_gap_tolerance=float("nan"),
        socp_tightness_margin=float("nan"),
        socp_tightness="not_applicable",
        socp_relaxation_tight=False,
        inverter_losses_kwh=0.0,
        socp_gap_max_relative_flow=float("nan"),
        socp_current_error_max_a=float("nan"),
        socp_loss_error_max_w=float("nan"),
    )
    case.formulation = "active_power_lindistflow"
    return case
