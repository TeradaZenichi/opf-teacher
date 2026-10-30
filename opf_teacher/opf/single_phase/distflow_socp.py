"""Single-phase DistFlow model with an SOCP relaxation."""
from __future__ import annotations

import math

import pyomo.environ as pyo

from opf_teacher.opf.single_phase.components import Case
from opf_teacher.opf.single_phase import pv_droop


THERMAL_LIMIT_NUMERICAL_MARGIN = 1e-10


class DistFlowModel:
    """Build network and device equations using the case's physical objects."""

    def __init__(self, case: Case):
        self.case = case
        self.base = case.base
        self.batteries = {battery.id: battery for battery in case.bess}
        self.pv = {pv.id: pv for pv in case.pv}
        self.branches = {
            bus_id: case.branches[case.parent_branch[bus_id]]
            for bus_id in case.buses if bus_id != case.root
        }
        self.voltage_upper_limit = pv_droop.relaxes_voltage(case)
        self.model = None

    def build(self) -> pyo.ConcreteModel:
        self.model = pyo.ConcreteModel(name=self.case.name)
        self.define_sets()
        self.define_variables()
        self.define_constraints()
        self.define_objective()
        return self.model

    def define_sets(self):
        model = self.model
        model.T = pyo.Set(initialize=self.case.periods, ordered=True)
        model.B = pyo.Set(initialize=list(self.case.buses))
        model.J = pyo.Set(initialize=list(self.branches), ordered=True)
        model.S = pyo.Set(initialize=list(self.batteries))
        model.G = pyo.Set(initialize=list(self.pv))

    def define_variables(self):
        model = self.model
        grid = self.case.grid
        model.v = pyo.Var(model.B, model.T, bounds=self.voltage_bounds)
        model.P = pyo.Var(model.J, model.T)
        model.Q = pyo.Var(model.J, model.T)
        model.l = pyo.Var(model.J, model.T, domain=pyo.NonNegativeReals, bounds=self.branch_current_bounds)
        model.pimp = pyo.Var(model.T, bounds=(0.0, self.base.pu_power(grid.p_import_max_kw)))
        model.pexp = pyo.Var(model.T, bounds=(0.0, self.base.pu_power(grid.p_export_max_kw)))
        reactive_limit = self.base.pu_power(grid.q_max_kvar)
        model.qgrid = pyo.Var(model.T, bounds=(-reactive_limit, reactive_limit))
        model.pch = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals, bounds=self.battery_charge_bounds)
        model.pdis = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals, bounds=self.battery_discharge_bounds)
        model.qbess = pyo.Var(model.S, model.T, bounds=self.battery_reactive_bounds)
        model.pbess_loss = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals, bounds=self.battery_loss_bounds)
        model.soc = pyo.Var(model.S, model.T, bounds=self.battery_energy_bounds)
        model.ppv = pyo.Var(model.G, model.T, domain=pyo.NonNegativeReals, bounds=self.pv_power_bounds)
        model.qpv = pyo.Var(model.G, model.T, bounds=self.pv_reactive_bounds)
        model.ppv_loss = pyo.Var(model.G, model.T, domain=pyo.NonNegativeReals, bounds=self.pv_loss_bounds)
        model.ppv_grid_consumption = pyo.Expression(model.G, model.T, rule=self.pv_grid_consumption_rule)
        model.ppv_net = pyo.Expression(model.G, model.T, rule=self.pv_net_power_rule)

    def voltage_bounds(self, model, bus_id, time):
        bus = self.case.buses[bus_id]
        if bus_id == self.case.root:
            fixed = self.case.grid.v_ref_pu ** 2
            return fixed, fixed
        upper = bus.v_max_pu if self.voltage_upper_limit is None else self.voltage_upper_limit
        return bus.v_min_pu ** 2, upper ** 2

    def branch_current_bounds(self, model, bus_id, time):
        branch = self.branches[bus_id]
        limit = self.base.pu_power(branch.s_max_kva) * (1.0 + THERMAL_LIMIT_NUMERICAL_MARGIN)
        return 0.0, limit ** 2

    def battery_charge_bounds(self, model, battery_id, time):
        battery = self.batteries[battery_id]
        return 0.0, self.base.pu_power(battery.p_charge_max_kw)

    def battery_discharge_bounds(self, model, battery_id, time):
        battery = self.batteries[battery_id]
        return 0.0, self.base.pu_power(battery.p_discharge_max_kw)

    def battery_reactive_bounds(self, model, battery_id, time):
        battery = self.batteries[battery_id]
        limit = self.base.pu_power(battery.s_max_kva) if battery.reactive_control else 0.0
        return -limit, limit

    def battery_loss_bounds(self, model, battery_id, time):
        battery = self.batteries[battery_id]
        upper = self.base.pu_power(battery.q_loss_rated_kw) if battery.reactive_control else 0.0
        return 0.0, upper

    def battery_energy_bounds(self, model, battery_id, time):
        battery = self.batteries[battery_id]
        capacity = self.base.pu_energy(battery.e_cap_kwh)
        return battery.soc_min_frac * capacity, battery.soc_max_frac * capacity

    def pv_power_bounds(self, model, pv_id, time):
        pv = self.pv[pv_id]
        available = self.base.pu_power(float(pv.avail_kw.iloc[time]))
        return 0.0, available

    def pv_reactive_bounds(self, model, pv_id, time):
        limit = self.base.pu_power(self.pv[pv_id].s_max_kva)
        return -limit, limit

    def pv_loss_bounds(self, model, pv_id, time):
        return 0.0, self.base.pu_power(self.pv[pv_id].q_loss_rated_kw)

    def pv_grid_consumption_rule(self, model, pv_id, time):
        pv = self.pv[pv_id]
        if pv.night_var and float(pv.avail_kw.iloc[time]) <= 0.0:
            return model.ppv_loss[pv_id, time]
        return 0.0

    def pv_net_power_rule(self, model, pv_id, time):
        return model.ppv[pv_id, time] - model.ppv_grid_consumption[pv_id, time]

    def active_injection(self, model, bus_id, time):
        bus = self.case.buses[bus_id]
        injection = -self.base.pu_power(float(bus.p_load_kw.iloc[time]))
        for battery in self.batteries.values():
            if battery.bus == bus_id:
                injection += model.pdis[battery.id, time] - model.pch[battery.id, time]
        for pv in self.pv.values():
            if pv.bus == bus_id:
                injection += model.ppv_net[pv.id, time]
        if bus_id == self.case.root:
            injection += model.pimp[time] - model.pexp[time]
        return injection

    def reactive_injection(self, model, bus_id, time):
        bus = self.case.buses[bus_id]
        injection = -self.base.pu_power(float(bus.q_load_kw.iloc[time]))
        for battery in self.batteries.values():
            if battery.bus == bus_id:
                injection += model.qbess[battery.id, time]
        for pv in self.pv.values():
            if pv.bus == bus_id:
                injection += model.qpv[pv.id, time]
        if bus_id == self.case.root:
            injection += model.qgrid[time]
        return injection

    def define_constraints(self):
        model = self.model
        model.balance_p = pyo.Constraint(model.B, model.T, rule=self.active_balance_rule)
        model.balance_q = pyo.Constraint(model.B, model.T, rule=self.reactive_balance_rule)
        model.voltage_drop = pyo.Constraint(model.J, model.T, rule=self.voltage_drop_rule)
        model.pv_capability = pyo.Constraint(model.G, model.T, rule=self.pv_capability_rule)
        model.pv_available_power = pyo.Constraint(model.G, model.T, rule=self.pv_available_power_rule)
        model.pv_q_loss = pyo.Constraint(model.G, model.T, rule=self.pv_loss_rule)
        model.pv_no_night_q = pyo.Constraint(model.G, model.T, rule=self.pv_night_reactive_rule)
        model.bess_capability = pyo.Constraint(model.S, model.T, rule=self.battery_capability_rule)
        model.bess_q_loss = pyo.Constraint(model.S, model.T, rule=self.battery_loss_rule)
        fixed = [(pv.id, time) for pv in self.pv.values()
                 if pv.control == "fixed_pf" for time in model.T]
        if fixed:
            model.pv_fixed_pf_idx = pyo.Set(initialize=fixed, dimen=2)
            model.pv_fixed_pf = pyo.Constraint(model.pv_fixed_pf_idx, rule=self.pv_power_factor_rule)
        pv_droop.PvDroopControl(self.case).add_to(model)
        model.socp = pyo.Constraint(model.J, model.T, rule=self.branch_cone_rule)
        model.soc_balance = pyo.Constraint(model.S, model.T, rule=self.battery_energy_rule)
        model.soc_terminal = pyo.Constraint(model.S, rule=self.battery_terminal_rule)
        model.no_roundtrip = pyo.SOSConstraint(model.T, rule=self.grid_exchange_rule, sos=1)

    def active_balance_rule(self, model, bus_id, time):
        outgoing = sum(model.P[child, time] for child in self.case.children.get(bus_id, []))
        incoming = 0.0
        if bus_id != self.case.root:
            branch = self.branches[bus_id]
            resistance, reactance = branch.impedance_pu(self.base, self.case.buses[branch.from_bus])
            incoming = model.P[bus_id, time] - resistance * model.l[bus_id, time]
        return incoming + self.active_injection(model, bus_id, time) == outgoing

    def reactive_balance_rule(self, model, bus_id, time):
        outgoing = sum(model.Q[child, time] for child in self.case.children.get(bus_id, []))
        incoming = 0.0
        if bus_id != self.case.root:
            branch = self.branches[bus_id]
            resistance, reactance = branch.impedance_pu(self.base, self.case.buses[branch.from_bus])
            incoming = model.Q[bus_id, time] - reactance * model.l[bus_id, time]
        return incoming + self.reactive_injection(model, bus_id, time) == outgoing

    def voltage_drop_rule(self, model, bus_id, time):
        branch = self.branches[bus_id]
        parent = self.case.buses[branch.from_bus]
        resistance, reactance = branch.impedance_pu(self.base, parent)
        return model.v[bus_id, time] == branch.tap_ratio ** 2 * (
            model.v[parent.id, time]
            - 2.0 * (resistance * model.P[bus_id, time] + reactance * model.Q[bus_id, time])
            + (resistance ** 2 + reactance ** 2) * model.l[bus_id, time]
        )

    def pv_power_factor_rule(self, model, pv_id, time):
        pv = self.pv[pv_id]
        ratio = math.tan(math.acos(pv.power_factor))
        return model.qpv[pv_id, time] == ratio * model.ppv[pv_id, time]

    def pv_capability_rule(self, model, pv_id, time):
        pv = self.pv[pv_id]
        limit = self.base.pu_power(pv.s_max_kva)
        return model.ppv[pv_id, time] ** 2 + model.qpv[pv_id, time] ** 2 <= limit ** 2

    def pv_available_power_rule(self, model, pv_id, time):
        pv = self.pv[pv_id]
        available = self.base.pu_power(float(pv.avail_kw.iloc[time]))
        solar_loss = model.ppv_loss[pv_id, time] - model.ppv_grid_consumption[pv_id, time]
        used = model.ppv[pv_id, time] + solar_loss
        if pv.curtailable or pv_droop.is_droop(pv.control):
            return used <= available
        return used == available

    def pv_loss_rule(self, model, pv_id, time):
        pv = self.pv[pv_id]
        rated_loss = self.base.pu_power(pv.q_loss_rated_kw)
        rating = self.base.pu_power(pv.s_max_kva)
        return rated_loss * model.qpv[pv_id, time] ** 2 <= rating ** 2 * model.ppv_loss[pv_id, time]

    def pv_night_reactive_rule(self, model, pv_id, time):
        pv = self.pv[pv_id]
        if float(pv.avail_kw.iloc[time]) > 0.0 or pv.night_var:
            return pyo.Constraint.Skip
        return model.qpv[pv_id, time] == 0.0

    def battery_capability_rule(self, model, battery_id, time):
        battery = self.batteries[battery_id]
        power = model.pdis[battery_id, time] - model.pch[battery_id, time]
        rating = self.base.pu_power(battery.s_max_kva)
        return power ** 2 + model.qbess[battery_id, time] ** 2 <= rating ** 2

    def battery_loss_rule(self, model, battery_id, time):
        battery = self.batteries[battery_id]
        rated_loss = self.base.pu_power(battery.q_loss_rated_kw)
        rating = self.base.pu_power(battery.s_max_kva)
        return rated_loss * model.qbess[battery_id, time] ** 2 <= rating ** 2 * model.pbess_loss[battery_id, time]

    def branch_cone_rule(self, model, bus_id, time):
        branch = self.branches[bus_id]
        return model.P[bus_id, time] ** 2 + model.Q[bus_id, time] ** 2 <= model.v[branch.from_bus, time] * model.l[bus_id, time]

    def battery_energy_rule(self, model, battery_id, time):
        battery = self.batteries[battery_id]
        gain = (battery.eta_charge * model.pch[battery_id, time]
                - model.pdis[battery_id, time] / battery.eta_discharge
                - model.pbess_loss[battery_id, time]) * self.case.dt_h
        if time == model.T.first():
            previous = battery.soc_init_frac * self.base.pu_energy(battery.e_cap_kwh)
        else:
            previous = model.soc[battery_id, model.T.prev(time)]
        return model.soc[battery_id, time] == previous + gain

    def battery_terminal_rule(self, model, battery_id):
        battery = self.batteries[battery_id]
        if not battery.cyclic_soc and battery.soc_terminal_frac is None:
            return pyo.Constraint.Skip
        target = battery.soc_terminal_frac
        if target is None:
            target = battery.soc_init_frac
        return model.soc[battery_id, model.T.last()] == target * self.base.pu_energy(battery.e_cap_kwh)

    def grid_exchange_rule(self, model, time):
        return [model.pimp[time], model.pexp[time]]

    def define_objective(self):
        model = self.model
        model.cost = pyo.Objective(rule=self.energy_cost_rule, sense=pyo.minimize)

    def energy_cost_rule(self, model):
        return self.base.s_base_kva * self.case.dt_h * sum(
            float(self.case.price.iloc[time]) * (model.pimp[time] - self.case.grid.feed_in_ratio * model.pexp[time])
            for time in model.T
        )


def build_model(case: Case) -> pyo.ConcreteModel:
    """Compatibility builder; new callers may use DistFlowModel directly."""
    return DistFlowModel(case).build()
