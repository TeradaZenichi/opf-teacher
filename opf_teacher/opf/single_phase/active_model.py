"""Single-phase active-power LinDistFlow model, without reactive power or losses."""
from __future__ import annotations

import math

import pandas as pd
import pyomo.environ as pyo

from opf_teacher.opf.common import Summary
from opf_teacher.opf.single_phase.components import BessResult, BranchResult, BusResult, Case, GridResult, PvResult


DEFAULT_TIE_BREAK_PENALTY = 1e-6


class ActivePowerModel:
    """Build LinDistFlow equations from the physical case components."""

    def __init__(self, case: Case, tie_break_penalty: float = DEFAULT_TIE_BREAK_PENALTY):
        if tie_break_penalty < 0.0:
            raise ValueError("tie_break_penalty cannot be negative")
        self.case = case
        self.base = case.base
        self.tie_break_penalty = tie_break_penalty
        self.batteries = {battery.id: battery for battery in case.bess}
        self.pv = {pv.id: pv for pv in case.pv}
        self.branches = {
            bus_id: case.branches[case.parent_branch[bus_id]]
            for bus_id in case.buses if bus_id != case.root
        }
        self.model = None

    def build(self) -> pyo.ConcreteModel:
        self.model = pyo.ConcreteModel(name=f"{self.case.name}-active-power")
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
        model.P = pyo.Var(model.J, model.T, bounds=self.branch_power_bounds)
        model.pimp = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, self.base.pu_power(grid.p_import_max_kw)))
        model.pexp = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, self.base.pu_power(grid.p_export_max_kw)))
        model.pch = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals, bounds=self.battery_charge_bounds)
        model.pdis = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals, bounds=self.battery_discharge_bounds)
        model.soc = pyo.Var(model.S, model.T, bounds=self.battery_energy_bounds)
        model.ppv = pyo.Var(model.G, model.T, domain=pyo.NonNegativeReals, bounds=self.pv_power_bounds)

    def voltage_bounds(self, model, bus_id, time):
        bus = self.case.buses[bus_id]
        if bus_id == self.case.root:
            fixed = self.case.grid.v_ref_pu ** 2
            return fixed, fixed
        return bus.v_min_pu ** 2, bus.v_max_pu ** 2

    def branch_power_bounds(self, model, bus_id, time):
        branch = self.branches[bus_id]
        limit = self.base.pu_power(branch.s_max_kva)
        return -limit, limit

    def battery_charge_bounds(self, model, battery_id, time):
        battery = self.batteries[battery_id]
        limit = min(battery.p_charge_max_kw, battery.s_max_kva)
        return 0.0, self.base.pu_power(limit)

    def battery_discharge_bounds(self, model, battery_id, time):
        battery = self.batteries[battery_id]
        limit = min(battery.p_discharge_max_kw, battery.s_max_kva)
        return 0.0, self.base.pu_power(limit)

    def battery_energy_bounds(self, model, battery_id, time):
        battery = self.batteries[battery_id]
        capacity = self.base.pu_energy(battery.e_cap_kwh)
        return battery.soc_min_frac * capacity, battery.soc_max_frac * capacity

    def pv_power_bounds(self, model, pv_id, time):
        pv = self.pv[pv_id]
        available = self.base.pu_power(float(pv.avail_kw.iloc[time]))
        upper = min(available, self.base.pu_power(pv.p_max_kw), self.base.pu_power(pv.s_max_kva))
        if pv.curtailable:
            return 0.0, upper
        return upper, upper

    def active_injection(self, model, bus_id, time):
        bus = self.case.buses[bus_id]
        injection = -self.base.pu_power(float(bus.p_load_kw.iloc[time]))
        for battery in self.batteries.values():
            if battery.bus == bus_id:
                injection += model.pdis[battery.id, time] - model.pch[battery.id, time]
        for pv in self.pv.values():
            if pv.bus == bus_id:
                injection += model.ppv[pv.id, time]
        if bus_id == self.case.root:
            injection += model.pimp[time] - model.pexp[time]
        return injection

    def define_constraints(self):
        model = self.model
        model.active_balance = pyo.Constraint(model.B, model.T, rule=self.active_balance_rule)
        model.voltage_drop = pyo.Constraint(model.J, model.T, rule=self.voltage_drop_rule)
        model.soc_balance = pyo.Constraint(model.S, model.T, rule=self.battery_energy_rule)
        model.soc_terminal = pyo.Constraint(model.S, rule=self.battery_terminal_rule)

    def active_balance_rule(self, model, bus_id, time):
        outgoing = sum(model.P[child, time] for child in self.case.children.get(bus_id, []))
        incoming = model.P[bus_id, time] if bus_id != self.case.root else 0.0
        return incoming + self.active_injection(model, bus_id, time) == outgoing

    def voltage_drop_rule(self, model, bus_id, time):
        branch = self.branches[bus_id]
        parent = self.case.buses[branch.from_bus]
        resistance, _ = branch.impedance_pu(self.base, parent)
        return model.v[bus_id, time] == branch.tap_ratio ** 2 * (
            model.v[parent.id, time] - 2.0 * resistance * model.P[bus_id, time]
        )

    def battery_energy_rule(self, model, battery_id, time):
        battery = self.batteries[battery_id]
        gain = (battery.eta_charge * model.pch[battery_id, time]
                - model.pdis[battery_id, time] / battery.eta_discharge) * self.case.dt_h
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

    def define_objective(self):
        model = self.model
        scale = self.base.s_base_kva * self.case.dt_h
        model.energy_cost = pyo.Expression(expr=scale * sum(
            float(self.case.price.iloc[time]) * (model.pimp[time] - self.case.grid.feed_in_ratio * model.pexp[time])
            for time in model.T
        ))
        model.tie_break_cost = pyo.Expression(expr=self.tie_break_penalty * scale * (
            sum(model.pimp[time] + model.pexp[time] for time in model.T)
            + sum(model.pch[battery_id, time] + model.pdis[battery_id, time]
                  for battery_id in model.S for time in model.T)
        ))
        model.cost = pyo.Objective(expr=model.energy_cost + model.tie_break_cost, sense=pyo.minimize)


def build_active_power_model(case: Case, tie_break_penalty: float = DEFAULT_TIE_BREAK_PENALTY) -> pyo.ConcreteModel:
    """Compatibility builder; new callers may use ActivePowerModel directly."""
    return ActivePowerModel(case, tie_break_penalty).build()


def solve_active_power_opf(
    case: Case,
    solver: str = "appsi_highs",
    tie_break_penalty: float = DEFAULT_TIE_BREAK_PENALTY,
) -> Case:
    """Compatibility API using the same solve path and private copy as Teacher."""
    # Local import avoids a cycle: Teacher also imports this model builder.
    from opf_teacher.teacher import SolveConfig, Teacher

    teacher = Teacher(case, formulation="single_phase_active")
    teacher.build(tie_break_penalty=tie_break_penalty)
    settings = SolveConfig(solver=solver)
    return teacher.solve(config=settings)


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
