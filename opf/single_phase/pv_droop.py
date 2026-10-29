"""Single-phase local Volt-VAr and Volt-Watt controls."""
import pyomo.environ as pyo

from opf.common import QMAX_FRAC, VV_V, VV_Q, VW_V, VW_P, V_MAX_EMERG


MODES = ("volt-var", "volt-watt", "volt-var-watt")


def is_droop(control):
    return control in MODES


def relaxes_voltage(case):
    return V_MAX_EMERG if any("watt" in g.control for g in case.pv) else None


def ppv_bounds(avail):
    return (0.0, avail)


class VoltageCurve:
    """Piecewise voltage curve represented by convex weights and an SOS2 set."""

    def __init__(self, pv_ids, periods, voltages):
        self.voltage_squared = [voltage ** 2 for voltage in voltages]
        self.indices = pyo.Set(initialize=[(pv_id, time) for pv_id in pv_ids for time in periods], dimen=2)
        self.points = pyo.Set(initialize=range(len(voltages)))
        self.weights = pyo.Var(self.indices, self.points, domain=pyo.NonNegativeReals)
        self.weight_sum = pyo.Constraint(self.indices, rule=self.weight_sum_rule)
        self.voltage_link = pyo.Constraint(self.indices, rule=self.voltage_link_rule)
        self.adjacent_weights = pyo.SOSConstraint(self.indices, rule=self.adjacent_weights_rule, sos=2)

    def weight_sum_rule(self, model, pv_id, time):
        return sum(self.weights[pv_id, time, point] for point in self.points) == 1

    def voltage_link_rule(self, model, pv_id, time):
        return model.vloc[pv_id, time] == sum(
            self.weights[pv_id, time, point] * self.voltage_squared[point] for point in self.points
        )

    def adjacent_weights_rule(self, model, pv_id, time):
        return [self.weights[pv_id, time, point] for point in self.points]


class PvDroopControl:
    """Add local Volt-VAr and Volt-Watt equations from PV components."""

    def __init__(self, case):
        self.case = case
        self.pv = {pv.id: pv for pv in case.pv}

    def add_to(self, model):
        var_ids = [pv.id for pv in self.pv.values() if "var" in pv.control]
        watt_ids = [pv.id for pv in self.pv.values() if "watt" in pv.control]
        if not (var_ids or watt_ids):
            return
        pure_watt = [(pv.id, time) for pv in self.pv.values()
                     if pv.control == "volt-watt" for time in model.T]
        if pure_watt:
            model.vw_noq_idx = pyo.Set(initialize=pure_watt, dimen=2)
            model.vw_noq = pyo.Constraint(model.vw_noq_idx, rule=self.no_reactive_power_rule)
        model.vloc = pyo.Var(model.G, model.T)
        model.vloc_link = pyo.Constraint(model.G, model.T, rule=self.local_voltage_rule)

        if var_ids:
            curve = VoltageCurve(var_ids, model.T, VV_V)
            model.vv_idx = curve.indices
            model.vv_k = curve.points
            model.vv_lam = curve.weights
            model.vv_sum = curve.weight_sum
            model.vv_x = curve.voltage_link
            model.vv_sos = curve.adjacent_weights
            model.volt_var = pyo.Constraint(model.vv_idx, rule=self.volt_var_rule)
        if watt_ids:
            curve = VoltageCurve(watt_ids, model.T, VW_V)
            model.vw_idx = curve.indices
            model.vw_k = curve.points
            model.vw_lam = curve.weights
            model.vw_sum = curve.weight_sum
            model.vw_x = curve.voltage_link
            model.vw_sos = curve.adjacent_weights
            model.volt_watt = pyo.Constraint(model.vw_idx, rule=self.volt_watt_rule)

    def no_reactive_power_rule(self, model, pv_id, time):
        return model.qpv[pv_id, time] == 0.0

    def local_voltage_rule(self, model, pv_id, time):
        pv = self.pv[pv_id]
        return model.vloc[pv_id, time] == model.v[pv.bus, time]

    def volt_var_rule(self, model, pv_id, time):
        pv = self.pv[pv_id]
        rating = self.case.base.pu_power(pv.s_max_kva)
        return model.qpv[pv_id, time] == sum(
            model.vv_lam[pv_id, time, point] * (QMAX_FRAC * rating * VV_Q[point]) for point in model.vv_k
        )

    def volt_watt_rule(self, model, pv_id, time):
        pv = self.pv[pv_id]
        available = self.case.base.pu_power(float(pv.avail_kw.iloc[time]))
        return model.ppv[pv_id, time] <= sum(
            model.vw_lam[pv_id, time, point] * (VW_P[point] * available) for point in model.vw_k
        )
