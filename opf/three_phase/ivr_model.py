"""Reduced-space unbalanced AC-IVR optimization.

The network state is solved in rectangular voltage/current coordinates for each
candidate dispatch. SciPy SLSQP optimizes the time-coupled BESS schedule while
the exact three-phase power flow enforces voltage and ampacity limits.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from opf.components import Summary
from opf.three_phase.components import Case
from opf.three_phase.results import (
    BessResult,
    BranchResult,
    BusResult,
    GridResult,
    PvResult,
)


@dataclass
class IVRSolution:
    voltages_kv: list[np.ndarray]
    branch_currents_ka: list[list[np.ndarray]]
    grid_power_kva: list[np.ndarray]
    p_charge_kw: np.ndarray
    p_discharge_kw: np.ndarray
    soc_kwh: np.ndarray
    objective_cost: float
    iterations: int
    max_power_flow_residual_a: float


class IVRModel:
    """Numerical representation of one phase-native AC-IVR problem."""

    def __init__(self, case: Case):
        if not isinstance(case, Case):
            raise TypeError("three_phase_ivr requires an opf.three_phase.Case")
        self.case = case
        self.solution: IVRSolution | None = None
        self._prepare_network()

    def _prepare_network(self):
        case = self.case
        self.nodes = [
            (bus_id, phase)
            for bus_id, bus in sorted(case.buses.items())
            for phase in bus.phases
        ]
        self.node_position = {node: position for position, node in enumerate(self.nodes)}
        self.slack_nodes = [(case.root, phase) for phase in case.grid.phases]
        self.slack_positions = [self.node_position[node] for node in self.slack_nodes]
        self.unknown_positions = [
            position for position in range(len(self.nodes))
            if position not in self.slack_positions
        ]

        voltage_bases = {
            round(float(bus.kv_base_ln), 9)
            for bus in case.buses.values()
            if bus.kv_base_ln is not None
        }
        if len(voltage_bases) != 1:
            raise ValueError("Three-phase IVR currently requires one line-neutral voltage level")
        self.v_base_ln_kv = voltage_bases.pop()

        count = len(self.nodes)
        self.ybus = np.zeros((count, count), dtype=complex)
        self.branch_admittances = []
        for branch in case.branches:
            impedance = np.asarray(branch.z_matrix_ohm, dtype=complex)
            try:
                admittance = np.linalg.inv(impedance)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"Branch {branch.name!r} impedance matrix is singular") from exc
            self.branch_admittances.append(admittance)
            from_positions = [self.node_position[(branch.from_bus, phase)] for phase in branch.phases]
            to_positions = [self.node_position[(branch.to_bus, phase)] for phase in branch.phases]
            for row, from_position in enumerate(from_positions):
                for column, other_from in enumerate(from_positions):
                    self.ybus[from_position, other_from] += admittance[row, column]
                for column, to_position in enumerate(to_positions):
                    self.ybus[from_position, to_position] -= admittance[row, column]
            for row, to_position in enumerate(to_positions):
                for column, other_to in enumerate(to_positions):
                    self.ybus[to_position, other_to] += admittance[row, column]
                for column, from_position in enumerate(from_positions):
                    self.ybus[to_position, from_position] -= admittance[row, column]

        unknown = np.ix_(self.unknown_positions, self.unknown_positions)
        coupling = np.ix_(self.unknown_positions, self.slack_positions)
        self.yuu = self.ybus[unknown]
        self.yus = self.ybus[coupling]
        if np.linalg.matrix_rank(self.yuu) != len(self.unknown_positions):
            raise ValueError("Three-phase network admittance matrix is singular or disconnected")

        self.slack_voltage = np.asarray([
            case.grid.v_ref_pu[phase]
            * self.v_base_ln_kv
            * np.exp(1j * math.radians(case.grid.angle_ref_deg[phase]))
            for phase in case.grid.phases
        ])
        self.nominal_voltage = np.zeros(count, dtype=complex)
        phase_reference = {
            phase: self.slack_voltage[index]
            for index, phase in enumerate(case.grid.phases)
        }
        for position, (_, phase) in enumerate(self.nodes):
            self.nominal_voltage[position] = phase_reference[phase]

        self.load_kva = np.zeros((case.n_periods, count), dtype=complex)
        for position, (bus_id, phase) in enumerate(self.nodes):
            bus = case.buses[bus_id]
            self.load_kva[:, position] = (
                bus.p_load_kw[phase].to_numpy()
                + 1j * bus.q_load_kvar[phase].to_numpy()
            )

        for connected in (*case.bess, *case.pv):
            if connected.connection.connection != "wye":
                raise ValueError("Three-phase IVR currently supports wye-connected devices")
            if connected.connection.dispatch_mode != "aggregate":
                raise ValueError("Three-phase IVR currently supports aggregate device dispatch")

    def solve(self, *, tee=False, maxiter=300, ftol=1e-8) -> Case:
        case = self.case
        periods = case.n_periods
        if len(case.bess) > 1:
            raise ValueError("Three-phase IVR currently supports at most one BESS")
        for connected in case.bess:
            if connected.device.reactive_control or connected.device.q_loss_rated_kw != 0.0:
                raise ValueError(
                    "Three-phase IVR currently requires BESS reactive_control=false "
                    "and q_loss_rated_kw=0"
                )
        for connected in case.pv:
            device = connected.device
            if (
                device.control != "fixed_pf"
                or device.curtailable
                or not math.isclose(device.power_factor, 1.0)
                or device.night_var
                or device.q_loss_rated_kw != 0.0
            ):
                raise ValueError(
                    "Three-phase IVR currently requires non-curtailable fixed_pf PV "
                    "at unity power factor, with night_var=false and q_loss_rated_kw=0"
                )

        bess = case.bess[0] if case.bess else None
        if bess is None:
            x0 = np.zeros(0)
            bounds = []
        else:
            bounds = (
                [(0.0, bess.device.p_charge_max_kw)] * periods
                + [(0.0, bess.device.p_discharge_max_kw)] * periods
            )
            x0 = self._economic_initial_dispatch(bounds)

        cache = {"key": None, "value": None}

        def evaluate(values):
            key = np.asarray(values, dtype=float).tobytes()
            if cache["key"] == key:
                return cache["value"]
            p_charge, p_discharge = self._split_dispatch(values)
            voltages = []
            branch_currents = []
            grid_power = []
            max_residual = 0.0
            previous_voltage = self.nominal_voltage
            for time in range(periods):
                consumption = self._net_consumption_kva(time, p_charge, p_discharge)
                voltage, currents, source_power, residual = self._power_flow(
                    consumption, previous_voltage
                )
                previous_voltage = voltage
                voltages.append(voltage)
                branch_currents.append(currents)
                grid_power.append(source_power)
                max_residual = max(max_residual, residual)

            total_grid_kw = np.asarray([power.real.sum() for power in grid_power])
            imported = np.maximum(total_grid_kw, 0.0)
            exported = np.maximum(-total_grid_kw, 0.0)
            objective = case.dt_h * np.sum(
                case.price.to_numpy()
                * (imported - case.grid.feed_in_ratio * exported)
            )
            objective += 1e-8 * case.dt_h * np.sum(p_charge + p_discharge)
            soc = self._soc_trajectory(p_charge, p_discharge)
            value = {
                "objective": float(objective),
                "voltages": voltages,
                "currents": branch_currents,
                "grid_power": grid_power,
                "soc": soc,
                "residual": max_residual,
            }
            cache["key"], cache["value"] = key, value
            return value

        constraints = []
        if bess is not None:
            constraints.extend([
                {"type": "eq", "fun": lambda values: self._terminal_soc_residual(values)},
                {"type": "ineq", "fun": lambda values: self._soc_margins(values)},
            ])
        constraints.append({
            "type": "ineq",
            "fun": lambda values: self._network_margins(evaluate(values)),
        })

        if x0.size:
            optimized = minimize(
                lambda values: evaluate(values)["objective"],
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": int(maxiter), "ftol": float(ftol), "disp": bool(tee)},
            )
            if not optimized.success:
                raise RuntimeError(f"Three-phase IVR optimization failed: {optimized.message}")
            values = optimized.x
            iterations = int(optimized.nit)
        else:
            values = x0
            margins = self._network_margins(evaluate(values))
            if margins.size and float(np.min(margins)) < -1e-7:
                raise RuntimeError("Fixed three-phase dispatch violates network limits")
            iterations = 0

        evaluated = evaluate(values)
        p_charge, p_discharge = self._split_dispatch(values)
        self.solution = IVRSolution(
            voltages_kv=evaluated["voltages"],
            branch_currents_ka=evaluated["currents"],
            grid_power_kva=evaluated["grid_power"],
            p_charge_kw=p_charge,
            p_discharge_kw=p_discharge,
            soc_kwh=evaluated["soc"],
            objective_cost=evaluated["objective"],
            iterations=iterations,
            max_power_flow_residual_a=evaluated["residual"] * 1000.0,
        )
        return attach_results(self, case)

    def _split_dispatch(self, values):
        periods = self.case.n_periods
        if not self.case.bess:
            return np.zeros(periods), np.zeros(periods)
        values = np.asarray(values, dtype=float)
        return values[:periods], values[periods:]

    def _economic_initial_dispatch(self, bounds):
        """Solve the lossless storage arbitrage LP to warm-start AC optimization."""
        periods = self.case.n_periods
        device = self.case.bess[0].device
        price = self.case.price.to_numpy() * self.case.dt_h
        objective = np.concatenate((price, -price))
        energy_coefficients = np.concatenate((
            np.full(periods, device.eta_charge * self.case.dt_h),
            np.full(periods, -self.case.dt_h / device.eta_discharge),
        ))
        rows = []
        upper = []
        initial = device.soc_init_frac * device.e_cap_kwh
        for time in range(periods):
            cumulative = np.zeros(2 * periods)
            cumulative[:time + 1] = energy_coefficients[:time + 1]
            cumulative[periods:periods + time + 1] = energy_coefficients[
                periods:periods + time + 1
            ]
            rows.extend((cumulative, -cumulative))
            upper.extend((
                device.soc_max_frac * device.e_cap_kwh - initial,
                initial - device.soc_min_frac * device.e_cap_kwh,
            ))
        target = (
            device.soc_init_frac
            if device.soc_terminal_frac is None
            else device.soc_terminal_frac
        ) * device.e_cap_kwh
        result = linprog(
            objective,
            A_ub=np.asarray(rows),
            b_ub=np.asarray(upper),
            A_eq=energy_coefficients.reshape(1, -1),
            b_eq=np.asarray([target - initial]),
            bounds=bounds,
            method="highs",
        )
        return result.x if result.success else np.zeros(2 * periods)

    def _net_consumption_kva(self, time, p_charge, p_discharge):
        consumption = self.load_kva[time].copy()
        for connected in self.case.bess:
            share = len(connected.connection.phases)
            p_net = p_charge[time] - p_discharge[time]
            for phase in connected.connection.phases:
                position = self.node_position[(connected.connection.bus, phase)]
                consumption[position] += p_net / share
        for connected in self.case.pv:
            for phase in connected.connection.phases:
                position = self.node_position[(connected.connection.bus, phase)]
                consumption[position] -= connected.available_kw[phase].iloc[time]
        return consumption

    def _power_flow(self, consumption_kva, initial_voltage):
        voltage = np.asarray(initial_voltage, dtype=complex).copy()
        voltage[self.slack_positions] = self.slack_voltage
        for _ in range(200):
            injection = np.zeros(len(self.nodes), dtype=complex)
            unknown_voltage = voltage[self.unknown_positions]
            if np.any(np.abs(unknown_voltage) < 1e-6):
                raise RuntimeError("Three-phase power flow reached a zero voltage")
            injection[self.unknown_positions] = -np.conj(
                (consumption_kva[self.unknown_positions] / 1000.0) / unknown_voltage
            )
            right_hand = injection[self.unknown_positions] - self.yus @ self.slack_voltage
            solved = np.linalg.solve(self.yuu, right_hand)
            updated = voltage.copy()
            updated[self.unknown_positions] = solved
            updated[self.slack_positions] = self.slack_voltage
            error = float(np.max(np.abs(updated - voltage)))
            voltage = 0.65 * updated + 0.35 * voltage
            if error < 1e-10:
                voltage = updated
                break
        else:
            raise RuntimeError("Three-phase power flow did not converge")

        currents = []
        for branch, admittance in zip(self.case.branches, self.branch_admittances):
            from_voltage = np.asarray([
                voltage[self.node_position[(branch.from_bus, phase)]]
                for phase in branch.phases
            ])
            to_voltage = np.asarray([
                voltage[self.node_position[(branch.to_bus, phase)]]
                for phase in branch.phases
            ])
            currents.append(admittance @ (from_voltage - to_voltage))

        grid_current = np.zeros(len(self.case.grid.phases), dtype=complex)
        grid_phase_position = {
            phase: position for position, phase in enumerate(self.case.grid.phases)
        }
        for branch, current in zip(self.case.branches, currents):
            if branch.from_bus == self.case.root:
                for phase_position, phase in enumerate(branch.phases):
                    grid_current[grid_phase_position[phase]] += current[phase_position]
            if branch.to_bus == self.case.root:
                for phase_position, phase in enumerate(branch.phases):
                    grid_current[grid_phase_position[phase]] -= current[phase_position]
        grid_power = self.slack_voltage * np.conj(grid_current) * 1000.0

        kcl = self.ybus @ voltage
        expected = np.zeros(len(self.nodes), dtype=complex)
        expected[self.unknown_positions] = -np.conj(
            (consumption_kva[self.unknown_positions] / 1000.0)
            / voltage[self.unknown_positions]
        )
        residual = float(np.max(np.abs(kcl[self.unknown_positions] - expected[self.unknown_positions])))
        return voltage, currents, grid_power, residual

    def _soc_trajectory(self, p_charge, p_discharge):
        if not self.case.bess:
            return np.zeros(self.case.n_periods)
        device = self.case.bess[0].device
        soc = np.empty(self.case.n_periods)
        previous = device.soc_init_frac * device.e_cap_kwh
        for time in range(self.case.n_periods):
            previous += self.case.dt_h * (
                device.eta_charge * p_charge[time]
                - p_discharge[time] / device.eta_discharge
            )
            soc[time] = previous
        return soc

    def _terminal_soc_residual(self, values):
        p_charge, p_discharge = self._split_dispatch(values)
        device = self.case.bess[0].device
        target_fraction = (
            device.soc_init_frac
            if device.soc_terminal_frac is None
            else device.soc_terminal_frac
        )
        return self._soc_trajectory(p_charge, p_discharge)[-1] - target_fraction * device.e_cap_kwh

    def _soc_margins(self, values):
        p_charge, p_discharge = self._split_dispatch(values)
        device = self.case.bess[0].device
        soc = self._soc_trajectory(p_charge, p_discharge)
        return np.concatenate((
            soc - device.soc_min_frac * device.e_cap_kwh,
            device.soc_max_frac * device.e_cap_kwh - soc,
        ))

    def _network_margins(self, evaluated):
        margins = []
        for voltage in evaluated["voltages"]:
            for position, (bus_id, phase) in enumerate(self.nodes):
                magnitude = abs(voltage[position]) / self.v_base_ln_kv
                bus = self.case.buses[bus_id]
                margins.extend((
                    magnitude - bus.v_min_pu[phase],
                    bus.v_max_pu[phase] - magnitude,
                ))
        for currents_at_time in evaluated["currents"]:
            for branch, current in zip(self.case.branches, currents_at_time):
                for position, phase in enumerate(branch.phases):
                    margins.append(branch.norm_amps[phase] - abs(current[position]) * 1000.0)
        for phase_power in evaluated["grid_power"]:
            active = float(phase_power.real.sum())
            reactive = float(phase_power.imag.sum())
            margins.extend((
                self.case.grid.p_import_max_kw - active,
                self.case.grid.p_export_max_kw + active,
                self.case.grid.q_max_kvar - reactive,
                self.case.grid.q_max_kvar + reactive,
            ))
        return np.asarray(margins)


def build_model(case: Case) -> IVRModel:
    return IVRModel(case)


def attach_results(model: IVRModel, case: Case) -> Case:
    solution = model.solution
    if solution is None:
        raise ValueError("The three-phase IVR model has not been solved")
    index = case.index

    for bus_id, bus in case.buses.items():
        voltage = {}
        angle = {}
        for phase in bus.phases:
            position = model.node_position[(bus_id, phase)]
            values = np.asarray([item[position] for item in solution.voltages_kv])
            voltage[phase] = pd.Series(abs(values) / model.v_base_ln_kv, index=index)
            angle[phase] = pd.Series(np.degrees(np.angle(values)), index=index)
        bus.result = BusResult(v_pu=voltage, angle_deg=angle)

    total_losses_kw = 0.0
    for branch_index, branch in enumerate(case.branches):
        p_values = {phase: [] for phase in branch.phases}
        q_values = {phase: [] for phase in branch.phases}
        i_values = {phase: [] for phase in branch.phases}
        loss_values = {phase: [] for phase in branch.phases}
        for time, voltage in enumerate(solution.voltages_kv):
            current = solution.branch_currents_ka[time][branch_index]
            from_voltage = np.asarray([
                voltage[model.node_position[(branch.from_bus, phase)]]
                for phase in branch.phases
            ])
            to_voltage = np.asarray([
                voltage[model.node_position[(branch.to_bus, phase)]]
                for phase in branch.phases
            ])
            sending = from_voltage * np.conj(current) * 1000.0
            receiving = to_voltage * np.conj(current) * 1000.0
            losses = sending.real - receiving.real
            for position, phase in enumerate(branch.phases):
                p_values[phase].append(sending[position].real)
                q_values[phase].append(sending[position].imag)
                i_values[phase].append(abs(current[position]) * 1000.0)
                loss_values[phase].append(losses[position])
        series = lambda values: {
            phase: pd.Series(items, index=index, dtype=float)
            for phase, items in values.items()
        }
        branch.result = BranchResult(
            p_kw=series(p_values),
            q_kvar=series(q_values),
            current_a=series(i_values),
            loss_kw=series(loss_values),
        )
        total_losses_kw += sum(float(values.sum()) for values in branch.result.loss_kw.values())

    grid_import = {phase: [] for phase in case.grid.phases}
    grid_export = {phase: [] for phase in case.grid.phases}
    grid_q = {phase: [] for phase in case.grid.phases}
    for values in solution.grid_power_kva:
        for position, phase in enumerate(case.grid.phases):
            grid_import[phase].append(max(values[position].real, 0.0))
            grid_export[phase].append(max(-values[position].real, 0.0))
            grid_q[phase].append(values[position].imag)
    to_series = lambda values: {
        phase: pd.Series(items, index=index, dtype=float)
        for phase, items in values.items()
    }
    import_series = to_series(grid_import)
    export_series = to_series(grid_export)
    cost = case.dt_h * case.price * (
        sum(import_series.values(), start=pd.Series(0.0, index=index))
        - case.grid.feed_in_ratio
        * sum(export_series.values(), start=pd.Series(0.0, index=index))
    )
    case.grid.result = GridResult(
        import_kw=import_series,
        export_kw=export_series,
        q_kvar=to_series(grid_q),
        cost=cost,
    )

    for connected in case.bess:
        phase_count = len(connected.connection.phases)
        p_net_total = solution.p_charge_kw - solution.p_discharge_kw
        connected.result = BessResult(
            p_net_kw={
                phase: pd.Series(p_net_total / phase_count, index=index)
                for phase in connected.connection.phases
            },
            q_injection_kvar={
                phase: pd.Series(0.0, index=index)
                for phase in connected.connection.phases
            },
            soc_kwh=pd.Series(solution.soc_kwh, index=index),
            soc_frac=pd.Series(solution.soc_kwh / connected.device.e_cap_kwh, index=index),
        )

    for connected in case.pv:
        generated = {phase: values.copy() for phase, values in connected.available_kw.items()}
        zeros = {phase: pd.Series(0.0, index=index) for phase in connected.connection.phases}
        connected.result = PvResult(
            available_kw=connected.available_kw,
            generation_kw=generated,
            q_injection_kvar=zeros,
            curtailment_kw={phase: values.copy() for phase, values in zeros.items()},
        )

    all_voltages = [
        value
        for bus in case.buses.values()
        for series in bus.result.v_pu.values()
        for value in series
    ]
    imported = case.grid.result.import_total_kw
    exported = case.grid.result.export_total_kw
    pv_generated = sum(
        float(series.sum())
        for connected in case.pv
        for series in connected.result.generation_kw.values()
    )
    case.summary = Summary(
        status="locally_optimal",
        objective_cost=solution.objective_cost,
        energy_cost=float(cost.sum()),
        energy_import_kwh=float(imported.sum()) * case.dt_h,
        energy_export_kwh=float(exported.sum()) * case.dt_h,
        pv_generated_kwh=pv_generated * case.dt_h,
        pv_curtailed_kwh=0.0,
        losses_kwh=total_losses_kw * case.dt_h,
        v_min_pu=min(all_voltages),
        v_max_pu=max(all_voltages),
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
    case.formulation = "three_phase_ac_ivr_reduced_space"
    case.quality = {
        "optimizer": "scipy_slsqp",
        "iterations": solution.iterations,
        "max_power_flow_residual_a": solution.max_power_flow_residual_a,
        "solution_quality": "local_feasible",
    }
    return case
