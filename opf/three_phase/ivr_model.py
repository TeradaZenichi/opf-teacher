"""Unbalanced three-phase AC optimization."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from opf.components import Summary
from opf.single_phase.pv_droop import (
    QMAX_FRAC,
    VV_Q,
    VV_V,
    VW_P,
    VW_V,
)
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
    q_bess_kvar: np.ndarray
    p_charge_phase_kw: np.ndarray
    p_discharge_phase_kw: np.ndarray
    q_bess_phase_kvar: np.ndarray
    soc_kwh: np.ndarray
    pv_generation_kw: np.ndarray
    q_pv_kvar: np.ndarray
    pv_generation_phase_kw: np.ndarray
    q_pv_phase_kvar: np.ndarray
    objective_cost: float
    iterations: int
    max_power_flow_residual_a: float


@dataclass
class Dispatch:
    p_charge: np.ndarray
    p_discharge: np.ndarray
    q_bess: np.ndarray
    p_pv: np.ndarray
    q_pv: np.ndarray
    p_charge_phase: np.ndarray
    p_discharge_phase: np.ndarray
    q_bess_phase: np.ndarray
    p_pv_phase: np.ndarray
    q_pv_phase: np.ndarray

    def copy(self):
        return Dispatch(**{name: values.copy() for name, values in vars(self).items()})


class IVRModel:
    """Three-phase AC-IVR problem."""

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
        self.phase_order = tuple(case.grid.phases)
        self.phase_position = {
            phase: position for position, phase in enumerate(self.phase_order)
        }
        self.slack_nodes = [(case.root, phase) for phase in case.grid.phases]
        self.slack_positions = [self.node_position[node] for node in self.slack_nodes]
        self.unknown_positions = [
            position for position in range(len(self.nodes))
            if position not in self.slack_positions
        ]

        if any(bus.kv_base_ln is None for bus in case.buses.values()):
            raise ValueError("Three-phase IVR requires a line-neutral voltage base at every bus")
        self.voltage_bases_kv = np.asarray([
            float(case.buses[bus_id].kv_base_ln)
            for bus_id, _ in self.nodes
        ])

        count = len(self.nodes)
        self.ybus = np.zeros((count, count), dtype=complex)
        self.branch_admittances = []
        self.branch_taps = []
        self.branch_from_connections = []
        self.branch_to_connections = []
        for branch in case.branches:
            impedance = np.asarray(branch.z_matrix_ohm, dtype=complex)
            tap = np.asarray(branch.tap_matrix, dtype=complex)
            from_connection, to_connection = (
                branch.connections or ("wye", "wye")
            )
            from_matrix = self._connection_matrix(
                from_connection, len(branch.phases)
            )
            to_matrix = self._connection_matrix(
                to_connection, len(branch.phases)
            )
            try:
                admittance = np.linalg.inv(impedance)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"Branch {branch.name!r} impedance matrix is singular") from exc
            self.branch_admittances.append(admittance)
            self.branch_taps.append(tap)
            self.branch_from_connections.append(from_matrix)
            self.branch_to_connections.append(to_matrix)
            from_positions = [self.node_position[(branch.from_bus, phase)] for phase in branch.phases]
            to_positions = [self.node_position[(branch.to_bus, phase)] for phase in branch.phases]
            yff = from_matrix.conj().T @ admittance @ from_matrix
            yft = -from_matrix.conj().T @ admittance @ tap @ to_matrix
            ytf = -to_matrix.conj().T @ tap.conj().T @ admittance @ from_matrix
            ytt = (
                to_matrix.conj().T @ tap.conj().T
                @ admittance @ tap @ to_matrix
            )
            for row, from_position in enumerate(from_positions):
                for column, other_from in enumerate(from_positions):
                    self.ybus[from_position, other_from] += yff[row, column]
                for column, to_position in enumerate(to_positions):
                    self.ybus[from_position, to_position] += yft[row, column]
            for row, to_position in enumerate(to_positions):
                for column, other_to in enumerate(to_positions):
                    self.ybus[to_position, other_to] += ytt[row, column]
                for column, from_position in enumerate(from_positions):
                    self.ybus[to_position, from_position] += ytf[row, column]

        unknown = np.ix_(self.unknown_positions, self.unknown_positions)
        coupling = np.ix_(self.unknown_positions, self.slack_positions)
        self.yuu = self.ybus[unknown]
        self.yus = self.ybus[coupling]
        if np.linalg.matrix_rank(self.yuu) != len(self.unknown_positions):
            raise ValueError("Three-phase network admittance matrix is singular or disconnected")

        self.slack_voltage = np.asarray([
            case.grid.v_ref_pu[phase]
            * self.voltage_bases_kv[self.node_position[(case.root, phase)]]
            * np.exp(1j * math.radians(case.grid.angle_ref_deg[phase]))
            for phase in case.grid.phases
        ])
        self.nominal_voltage = np.zeros(count, dtype=complex)
        phase_angle = {
            phase: np.exp(1j * math.radians(case.grid.angle_ref_deg[phase]))
            for phase in case.grid.phases
        }
        for position, (_, phase) in enumerate(self.nodes):
            self.nominal_voltage[position] = self.voltage_bases_kv[position] * phase_angle[phase]

        self.load_kva = np.zeros((case.n_periods, count), dtype=complex)
        for position, (bus_id, phase) in enumerate(self.nodes):
            bus = case.buses[bus_id]
            self.load_kva[:, position] = (
                bus.p_load_kw[phase].to_numpy()
                + 1j * bus.q_load_kvar[phase].to_numpy()
            )

        self.pv_available_total = [
            connected.available_total_kw.to_numpy() for connected in case.pv
        ]

        for connected in (*case.bess, *case.pv):
            if (
                connected.connection.connection == "delta"
                and len(connected.connection.phases) != 3
            ):
                raise ValueError(
                    "Three-phase IVR delta devices must define phases a, b and c"
                )

    @staticmethod
    def _connection_matrix(connection, size):
        if connection == "wye":
            return np.eye(size, dtype=complex)
        if connection == "delta" and size == 3:
            return np.asarray(
                ((1.0, 0.0, -1.0), (-1.0, 1.0, 0.0), (0.0, -1.0, 1.0)),
                dtype=complex,
            )
        raise ValueError("Delta branches require exactly three phases")

    def solve(self, *, tee=False, maxiter=300, ftol=1e-8) -> Case:
        case = self.case
        periods = case.n_periods
        supported_controls = {
            "optimal", "fixed_pf", "volt-var", "volt-watt", "volt-var-watt"
        }
        for connected in case.pv:
            if connected.device.control not in supported_controls:
                raise ValueError(
                    f"Unsupported three-phase PV control {connected.device.control!r}"
                )

        bounds, x0 = self._decision_bounds_and_start()

        cache = {"key": None, "value": None}

        def evaluate(values):
            key = np.asarray(values, dtype=float).tobytes()
            if cache["key"] == key:
                return cache["value"]
            dispatch = self._unpack_dispatch(values)
            voltages = []
            branch_currents = []
            grid_power = []
            max_residual = 0.0
            previous_voltage = self.nominal_voltage
            for time in range(periods):
                consumption = self._net_consumption_kva(time, dispatch)
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
            objective += 1e-8 * case.dt_h * np.sum(
                dispatch.p_charge + dispatch.p_discharge
            )
            soc = self._soc_trajectory(dispatch)
            value = {
                "objective": float(objective),
                "voltages": voltages,
                "currents": branch_currents,
                "grid_power": grid_power,
                "soc": soc,
                "dispatch": dispatch,
                "residual": max_residual,
            }
            cache["key"], cache["value"] = key, value
            return value

        constraints = []
        if self._has_device_equalities():
            constraints.append({
                "type": "eq",
                "fun": lambda values: self._device_equalities(values, evaluate(values)),
            })
        if case.bess or case.pv:
            constraints.append({
                "type": "ineq",
                "fun": lambda values: self._device_margins(values, evaluate(values)),
            })
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
        dispatch = self._unpack_dispatch(values)
        self.solution = IVRSolution(
            voltages_kv=evaluated["voltages"],
            branch_currents_ka=evaluated["currents"],
            grid_power_kva=evaluated["grid_power"],
            p_charge_kw=dispatch.p_charge,
            p_discharge_kw=dispatch.p_discharge,
            q_bess_kvar=dispatch.q_bess,
            p_charge_phase_kw=dispatch.p_charge_phase,
            p_discharge_phase_kw=dispatch.p_discharge_phase,
            q_bess_phase_kvar=dispatch.q_bess_phase,
            soc_kwh=evaluated["soc"],
            pv_generation_kw=dispatch.p_pv,
            q_pv_kvar=dispatch.q_pv,
            pv_generation_phase_kw=dispatch.p_pv_phase,
            q_pv_phase_kvar=dispatch.q_pv_phase,
            objective_cost=evaluated["objective"],
            iterations=iterations,
            max_power_flow_residual_a=evaluated["residual"] * 1000.0,
        )
        return attach_results(self, case)

    def _decision_bounds_and_start(self):
        periods = self.case.n_periods
        bess_count = len(self.case.bess)
        pv_count = len(self.case.pv)
        phase_count = len(self.phase_order)
        dispatch = Dispatch(
            p_charge=np.zeros((bess_count, periods)),
            p_discharge=np.zeros((bess_count, periods)),
            q_bess=np.zeros((bess_count, periods)),
            p_pv=np.zeros((pv_count, periods)),
            q_pv=np.zeros((pv_count, periods)),
            p_charge_phase=np.zeros((bess_count, phase_count, periods)),
            p_discharge_phase=np.zeros((bess_count, phase_count, periods)),
            q_bess_phase=np.zeros((bess_count, phase_count, periods)),
            p_pv_phase=np.zeros((pv_count, phase_count, periods)),
            q_pv_phase=np.zeros((pv_count, phase_count, periods)),
        )
        self._variable_layout = []
        bounds = []
        start = []

        def add(name, indices, lower, upper, initial):
            self._variable_layout.append((name, tuple(indices)))
            bounds.append((float(lower), float(upper)))
            start.append(float(np.clip(initial, lower, upper)))

        for index, connected in enumerate(self.case.bess):
            device = connected.device
            device_bounds = (
                [(0.0, device.p_charge_max_kw)] * periods
                + [(0.0, device.p_discharge_max_kw)] * periods
            )
            initial = self._economic_initial_dispatch(connected, device_bounds)
            if connected.connection.dispatch_mode == "aggregate":
                for time in range(periods):
                    add(
                        "p_charge", (index, time), 0.0,
                        device.p_charge_max_kw, initial[time],
                    )
                for time in range(periods):
                    add(
                        "p_discharge", (index, time), 0.0,
                        device.p_discharge_max_kw, initial[periods + time],
                    )
                if device.reactive_control:
                    for time in range(periods):
                        add(
                            "q_bess", (index, time), -device.s_max_kva,
                            device.s_max_kva, 0.0,
                        )
            else:
                count = len(connected.connection.phases)
                for phase in connected.connection.phases:
                    phase_index = self.phase_position[phase]
                    for time in range(periods):
                        add(
                            "p_charge_phase", (index, phase_index, time),
                            0.0, device.p_charge_max_kw,
                            initial[time] / count,
                        )
                    for time in range(periods):
                        add(
                            "p_discharge_phase", (index, phase_index, time),
                            0.0, device.p_discharge_max_kw,
                            initial[periods + time] / count,
                        )
                    if device.reactive_control:
                        for time in range(periods):
                            add(
                                "q_bess_phase", (index, phase_index, time),
                                -device.s_max_kva / count,
                                device.s_max_kva / count,
                                0.0,
                            )

        for index, connected in enumerate(self.case.pv):
            device = connected.device
            available = np.minimum(
                connected.available_total_kw.to_numpy(), device.p_max_kw
            )
            dispatch.p_pv[index] = available
            if "var" in device.control:
                initial_voltage = float(np.mean([
                    self.case.grid.v_ref_pu[phase]
                    for phase in connected.connection.phases
                ]))
                dispatch.q_pv[index] = QMAX_FRAC * device.s_max_kva * np.interp(
                    initial_voltage,
                    VV_V,
                    VV_Q,
                )
            if "watt" in device.control:
                initial_voltage = float(np.mean([
                    self.case.grid.v_ref_pu[phase]
                    for phase in connected.connection.phases
                ]))
                fraction = np.interp(
                    initial_voltage,
                    VW_V,
                    VW_P,
                )
                dispatch.p_pv[index] *= fraction
            for time, value in enumerate(available):
                p_is_variable = (
                    device.curtailable
                    or device.q_loss_rated_kw > 0.0
                    or "watt" in device.control
                )
                if p_is_variable and connected.connection.dispatch_mode == "aggregate":
                    add(
                        "p_pv", (index, time), 0.0, value,
                        dispatch.p_pv[index, time],
                    )
            for time, value in enumerate(available):
                q_enabled = (
                    device.control in {"optimal", "volt-var", "volt-var-watt"}
                    and (value > 0.0 or device.night_var)
                )
                if q_enabled and connected.connection.dispatch_mode == "aggregate":
                    add(
                        "q_pv", (index, time), -device.s_max_kva,
                        device.s_max_kva, dispatch.q_pv[index, time],
                    )

            if connected.connection.dispatch_mode == "per_phase":
                count = len(connected.connection.phases)
                for phase in connected.connection.phases:
                    phase_index = self.phase_position[phase]
                    phase_available = np.minimum(
                        connected.available_kw[phase].to_numpy(),
                        device.p_max_kw / count,
                    )
                    dispatch.p_pv_phase[index, phase_index] = phase_available
                    for time, value in enumerate(phase_available):
                        p_is_variable = (
                            device.curtailable
                            or device.q_loss_rated_kw > 0.0
                            or "watt" in device.control
                        )
                        if p_is_variable:
                            add(
                                "p_pv_phase", (index, phase_index, time),
                                0.0, value, value,
                            )
                        q_enabled = (
                            device.control in {
                                "optimal", "volt-var", "volt-var-watt"
                            }
                            and (value > 0.0 or device.night_var)
                        )
                        if q_enabled:
                            add(
                                "q_pv_phase", (index, phase_index, time),
                                -device.s_max_kva / count,
                                device.s_max_kva / count,
                                0.0,
                            )

        self._fixed_dispatch = dispatch
        self._variable_keys = set(self._variable_layout)
        return bounds, np.asarray(start, dtype=float)

    def _unpack_dispatch(self, values):
        dispatch = self._fixed_dispatch.copy()
        for value, (name, indices) in zip(
            np.asarray(values, dtype=float), self._variable_layout
        ):
            getattr(dispatch, name)[indices] = value
        for index, connected in enumerate(self.case.bess):
            if connected.connection.dispatch_mode == "aggregate":
                count = len(connected.connection.phases)
                for phase in connected.connection.phases:
                    phase_index = self.phase_position[phase]
                    dispatch.p_charge_phase[index, phase_index] = dispatch.p_charge[index] / count
                    dispatch.p_discharge_phase[index, phase_index] = dispatch.p_discharge[index] / count
                    dispatch.q_bess_phase[index, phase_index] = dispatch.q_bess[index] / count
            else:
                phase_indices = [
                    self.phase_position[phase]
                    for phase in connected.connection.phases
                ]
                dispatch.p_charge[index] = dispatch.p_charge_phase[index, phase_indices].sum(axis=0)
                dispatch.p_discharge[index] = dispatch.p_discharge_phase[index, phase_indices].sum(axis=0)
                dispatch.q_bess[index] = dispatch.q_bess_phase[index, phase_indices].sum(axis=0)
        for index, connected in enumerate(self.case.pv):
            if connected.connection.dispatch_mode == "aggregate":
                count = len(connected.connection.phases)
                for phase in connected.connection.phases:
                    phase_index = self.phase_position[phase]
                    dispatch.p_pv_phase[index, phase_index] = dispatch.p_pv[index] / count
                    dispatch.q_pv_phase[index, phase_index] = dispatch.q_pv[index] / count
            else:
                phase_indices = [
                    self.phase_position[phase]
                    for phase in connected.connection.phases
                ]
                dispatch.p_pv[index] = dispatch.p_pv_phase[index, phase_indices].sum(axis=0)
                dispatch.q_pv[index] = dispatch.q_pv_phase[index, phase_indices].sum(axis=0)
            if connected.device.control == "fixed_pf":
                ratio = math.tan(math.acos(connected.device.power_factor))
                if connected.connection.dispatch_mode == "aggregate":
                    dispatch.q_pv[index] = ratio * dispatch.p_pv[index]
                    for phase in connected.connection.phases:
                        phase_index = self.phase_position[phase]
                        dispatch.q_pv_phase[index, phase_index] = ratio * dispatch.p_pv_phase[index, phase_index]
                else:
                    for phase in connected.connection.phases:
                        phase_index = self.phase_position[phase]
                        dispatch.q_pv_phase[index, phase_index] = ratio * dispatch.p_pv_phase[index, phase_index]
                    dispatch.q_pv[index] = dispatch.q_pv_phase[index, phase_indices].sum(axis=0)
        return dispatch

    def _economic_initial_dispatch(self, connected, bounds):
        """Solve the lossless storage arbitrage LP to warm-start AC optimization."""
        periods = self.case.n_periods
        device = connected.device
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
        has_terminal_target = (
            device.cyclic_soc or device.soc_terminal_frac is not None
        )
        target = (
            device.soc_init_frac
            if device.soc_terminal_frac is None
            else device.soc_terminal_frac
        ) * device.e_cap_kwh
        result = linprog(
            objective,
            A_ub=np.asarray(rows),
            b_ub=np.asarray(upper),
            A_eq=(
                energy_coefficients.reshape(1, -1)
                if has_terminal_target else None
            ),
            b_eq=(
                np.asarray([target - initial])
                if has_terminal_target else None
            ),
            bounds=bounds,
            method="highs",
        )
        return result.x if result.success else np.zeros(2 * periods)

    def _net_consumption_kva(self, time, dispatch):
        consumption = self.load_kva[time].copy()
        delta_consumption = []
        for index, connected in enumerate(self.case.bess):
            delta_values = []
            for phase in connected.connection.phases:
                phase_index = self.phase_position[phase]
                position = self.node_position[(connected.connection.bus, phase)]
                p_net = dispatch.p_charge_phase[index, phase_index, time] - dispatch.p_discharge_phase[index, phase_index, time]
                q_injection = dispatch.q_bess_phase[index, phase_index, time]
                power = p_net - 1j * q_injection
                if connected.connection.connection == "wye":
                    consumption[position] += power
                else:
                    delta_values.append(power)
            if delta_values:
                delta_consumption.append((connected.connection, delta_values))
        for index, connected in enumerate(self.case.pv):
            available = self.pv_available_total[index][time]
            delta_values = []
            for phase in connected.connection.phases:
                phase_index = self.phase_position[phase]
                position = self.node_position[(connected.connection.bus, phase)]
                generated = dispatch.p_pv_phase[index, phase_index, time]
                q_injection = dispatch.q_pv_phase[index, phase_index, time]
                loss = self._pv_phase_q_loss(
                    index, phase_index, q_injection
                )
                p_net = generated - (loss if available <= 0.0 else 0.0)
                power = -p_net - 1j * q_injection
                if connected.connection.connection == "wye":
                    consumption[position] += power
                else:
                    delta_values.append(power)
            if delta_values:
                delta_consumption.append((connected.connection, delta_values))
        return consumption, delta_consumption

    def _current_injection(self, consumption_kva, delta_consumption, voltage):
        injection = np.zeros(len(self.nodes), dtype=complex)
        nonzero = np.abs(voltage) >= 1e-6
        injection[nonzero] = -np.conj(
            (consumption_kva[nonzero] / 1000.0) / voltage[nonzero]
        )
        for connection, powers in delta_consumption:
            phases = connection.phases
            for index, (phase, power) in enumerate(zip(phases, powers)):
                next_phase = phases[(index - 1) % len(phases)]
                first = self.node_position[(connection.bus, phase)]
                second = self.node_position[(connection.bus, next_phase)]
                line_voltage = voltage[first] - voltage[second]
                if abs(line_voltage) < 1e-6:
                    raise RuntimeError("Three-phase power flow reached a zero delta voltage")
                leg_current = np.conj((power / 1000.0) / line_voltage)
                injection[first] -= leg_current
                injection[second] += leg_current
        return injection

    def _power_flow(self, consumption_data, initial_voltage):
        consumption_kva, delta_consumption = consumption_data
        voltage = np.asarray(initial_voltage, dtype=complex).copy()
        voltage[self.slack_positions] = self.slack_voltage
        for _ in range(200):
            unknown_voltage = voltage[self.unknown_positions]
            if np.any(np.abs(unknown_voltage) < 1e-6):
                raise RuntimeError("Three-phase power flow reached a zero voltage")
            injection = self._current_injection(
                consumption_kva, delta_consumption, voltage
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
        for branch, admittance, tap, from_matrix, to_matrix in zip(
            self.case.branches,
            self.branch_admittances,
            self.branch_taps,
            self.branch_from_connections,
            self.branch_to_connections,
        ):
            from_voltage = np.asarray([
                voltage[self.node_position[(branch.from_bus, phase)]]
                for phase in branch.phases
            ])
            to_voltage = np.asarray([
                voltage[self.node_position[(branch.to_bus, phase)]]
                for phase in branch.phases
            ])
            currents.append(
                admittance @ (from_matrix @ from_voltage - tap @ to_matrix @ to_voltage)
            )

        injection = self._current_injection(
            consumption_kva, delta_consumption, voltage
        )
        grid_current = (self.ybus @ voltage - injection)[self.slack_positions]
        grid_power = self.slack_voltage * np.conj(grid_current) * 1000.0

        kcl = self.ybus @ voltage
        expected = injection
        residual = float(np.max(np.abs(kcl[self.unknown_positions] - expected[self.unknown_positions])))
        return voltage, currents, grid_power, residual

    def _soc_trajectory(self, dispatch):
        soc = np.empty((len(self.case.bess), self.case.n_periods))
        for index, connected in enumerate(self.case.bess):
            device = connected.device
            previous = device.soc_init_frac * device.e_cap_kwh
            for time in range(self.case.n_periods):
                previous += self.case.dt_h * (
                    device.eta_charge * dispatch.p_charge[index, time]
                    - dispatch.p_discharge[index, time] / device.eta_discharge
                    - self._bess_dispatch_q_loss(index, dispatch, time)
                )
                soc[index, time] = previous
        return soc

    def _has_device_equalities(self):
        if any(
            connected.device.cyclic_soc
            or connected.device.soc_terminal_frac is not None
            for connected in self.case.bess
        ):
            return True
        return any(
            (
                not connected.device.curtailable
                and connected.device.q_loss_rated_kw > 0.0
            )
            or "var" in connected.device.control
            for connected in self.case.pv
        )

    def _device_equalities(self, values, evaluated):
        dispatch = evaluated["dispatch"]
        equalities = []
        for index, connected in enumerate(self.case.bess):
            device = connected.device
            if device.cyclic_soc or device.soc_terminal_frac is not None:
                target = (
                    device.soc_init_frac
                    if device.soc_terminal_frac is None
                    else device.soc_terminal_frac
                ) * device.e_cap_kwh
                equalities.append(evaluated["soc"][index, -1] - target)
        for index, connected in enumerate(self.case.pv):
            device = connected.device
            available = connected.available_total_kw.to_numpy()
            for time in range(self.case.n_periods):
                p_value = dispatch.p_pv[index, time]
                q_value = dispatch.q_pv[index, time]
                loss = self._pv_dispatch_q_loss(index, dispatch, time)
                if not device.curtailable and device.q_loss_rated_kw > 0.0:
                    if connected.connection.dispatch_mode == "aggregate":
                        equalities.append(
                            p_value + (loss if available[time] > 0.0 else 0.0)
                            - min(available[time], device.p_max_kw)
                        )
                    else:
                        for phase in connected.connection.phases:
                            phase_index = self.phase_position[phase]
                            phase_available = min(
                                connected.available_kw[phase].iloc[time],
                                device.p_max_kw
                                / len(connected.connection.phases),
                            )
                            phase_q = dispatch.q_pv_phase[index, phase_index, time]
                            phase_loss = self._pv_phase_q_loss(
                                index, phase_index, phase_q
                            )
                            equalities.append(
                                dispatch.p_pv_phase[index, phase_index, time]
                                + (
                                    phase_loss
                                    if phase_available > 0.0 else 0.0
                                )
                                - phase_available
                            )
                if "var" in device.control:
                    if connected.connection.dispatch_mode == "aggregate":
                        voltage = self._device_voltage_pu(
                            connected, evaluated["voltages"][time]
                        )
                        target = QMAX_FRAC * device.s_max_kva * np.interp(
                            voltage, VV_V, VV_Q
                        )
                        equalities.append(q_value - target)
                    else:
                        count = len(connected.connection.phases)
                        for phase in connected.connection.phases:
                            phase_index = self.phase_position[phase]
                            voltage = self._device_phase_voltage_pu(
                                connected, phase, evaluated["voltages"][time]
                            )
                            target = (
                                QMAX_FRAC * device.s_max_kva / count
                                * np.interp(voltage, VV_V, VV_Q)
                            )
                            equalities.append(
                                dispatch.q_pv_phase[index, phase_index, time] - target
                            )
        return np.asarray(equalities, dtype=float)

    def _device_margins(self, values, evaluated):
        dispatch = evaluated["dispatch"]
        margins = []
        for index, connected in enumerate(self.case.bess):
            device = connected.device
            soc = evaluated["soc"][index]
            margins.extend(soc - device.soc_min_frac * device.e_cap_kwh)
            margins.extend(device.soc_max_frac * device.e_cap_kwh - soc)
            if connected.connection.dispatch_mode == "per_phase":
                count = len(connected.connection.phases)
                phase_indices = [
                    self.phase_position[phase]
                    for phase in connected.connection.phases
                ]
                margins.extend(
                    device.p_charge_max_kw
                    - dispatch.p_charge_phase[index, phase_indices].sum(axis=0)
                )
                margins.extend(
                    device.p_discharge_max_kw
                    - dispatch.p_discharge_phase[index, phase_indices].sum(axis=0)
                )
                for phase in connected.connection.phases:
                    phase_index = self.phase_position[phase]
                    p_net = dispatch.p_charge_phase[index, phase_index] - dispatch.p_discharge_phase[index, phase_index]
                    q_value = dispatch.q_bess_phase[index, phase_index]
                    margins.extend(
                        (device.s_max_kva / count) ** 2
                        - p_net ** 2 - q_value ** 2
                    )
            elif (
                device.reactive_control
                or device.p_charge_max_kw > device.s_max_kva
                or device.p_discharge_max_kw > device.s_max_kva
            ):
                p_net = dispatch.p_charge[index] - dispatch.p_discharge[index]
                q_value = dispatch.q_bess[index]
                margins.extend(
                    device.s_max_kva ** 2 - p_net ** 2 - q_value ** 2
                )
        for index, connected in enumerate(self.case.pv):
            device = connected.device
            variable = any(
                key[1][0] == index and key[0] in {
                    "p_pv", "q_pv", "p_pv_phase", "q_pv_phase"
                }
                for key in self._variable_keys
            )
            if not variable:
                continue
            available = np.minimum(
                connected.available_total_kw.to_numpy(), device.p_max_kw
            )
            for time in range(self.case.n_periods):
                if connected.connection.dispatch_mode == "per_phase":
                    count = len(connected.connection.phases)
                    for phase in connected.connection.phases:
                        phase_index = self.phase_position[phase]
                        phase_available = min(
                            connected.available_kw[phase].iloc[time],
                            device.p_max_kw / count,
                        )
                        phase_p = dispatch.p_pv_phase[index, phase_index, time]
                        phase_q = dispatch.q_pv_phase[index, phase_index, time]
                        phase_loss = self._pv_phase_q_loss(
                            index, phase_index, phase_q
                        )
                        margins.extend((
                            (device.s_max_kva / count) ** 2
                            - phase_p ** 2 - phase_q ** 2,
                            phase_available - phase_p - (
                                phase_loss if phase_available > 0.0 else 0.0
                            ),
                        ))
                        if "watt" in device.control:
                            voltage = self._device_phase_voltage_pu(
                                connected, phase,
                                evaluated["voltages"][time],
                            )
                            fraction = np.interp(voltage, VW_V, VW_P)
                            margins.append(
                                fraction * phase_available - phase_p
                            )
                    continue
                p_value = dispatch.p_pv[index, time]
                q_value = dispatch.q_pv[index, time]
                loss = self._pv_dispatch_q_loss(index, dispatch, time)
                margins.extend((
                    device.s_max_kva ** 2 - p_value ** 2 - q_value ** 2,
                    available[time] - p_value - (
                        loss if available[time] > 0.0 else 0.0
                    ),
                ))
                if "watt" in device.control:
                    voltage = self._device_voltage_pu(
                        connected, evaluated["voltages"][time]
                    )
                    fraction = np.interp(
                        voltage,
                        VW_V,
                        VW_P,
                    )
                    margins.append(fraction * available[time] - p_value)
        return np.asarray(margins, dtype=float)

    def _bess_q_loss(self, index, q_value):
        device = self.case.bess[index].device
        if not device.reactive_control or device.q_loss_rated_kw <= 0.0:
            return 0.0
        if np.ndim(q_value):
            return (
                device.q_loss_rated_kw
                * len(q_value)
                * np.sum((np.asarray(q_value) / device.s_max_kva) ** 2)
            )
        return device.q_loss_rated_kw * (q_value / device.s_max_kva) ** 2

    def _bess_dispatch_q_loss(self, index, dispatch, time):
        connected = self.case.bess[index]
        if connected.connection.dispatch_mode == "aggregate":
            return self._bess_q_loss(
                index, dispatch.q_bess[index, time]
            )
        phase_indices = [
            self.phase_position[phase]
            for phase in connected.connection.phases
        ]
        return self._bess_q_loss(
            index,
            dispatch.q_bess_phase[index, phase_indices, time],
        )

    def _pv_q_loss(self, index, q_value):
        device = self.case.pv[index].device
        if device.q_loss_rated_kw <= 0.0:
            return 0.0
        if np.ndim(q_value):
            return (
                device.q_loss_rated_kw
                * len(q_value)
                * np.sum((np.asarray(q_value) / device.s_max_kva) ** 2)
            )
        return device.q_loss_rated_kw * (q_value / device.s_max_kva) ** 2

    def _pv_dispatch_q_loss(self, index, dispatch, time):
        connected = self.case.pv[index]
        if connected.connection.dispatch_mode == "aggregate":
            return self._pv_q_loss(index, dispatch.q_pv[index, time])
        phase_indices = [
            self.phase_position[phase]
            for phase in connected.connection.phases
        ]
        return self._pv_q_loss(
            index,
            dispatch.q_pv_phase[index, phase_indices, time],
        )

    def _pv_phase_q_loss(self, index, phase_index, q_value):
        device = self.case.pv[index].device
        if device.q_loss_rated_kw <= 0.0:
            return 0.0
        return (
            device.q_loss_rated_kw
            * len(self.case.pv[index].connection.phases)
            * (q_value / device.s_max_kva) ** 2
        )

    def _device_voltage_pu(self, connected, voltage):
        values = [
            self._device_phase_voltage_pu(connected, phase, voltage)
            for phase in connected.connection.phases
        ]
        return float(np.mean(values))

    def _device_phase_voltage_pu(self, connected, phase, voltage):
        position = self.node_position[(connected.connection.bus, phase)]
        if connected.connection.connection == "delta":
            phases = connected.connection.phases
            phase_index = phases.index(phase)
            next_phase = phases[(phase_index - 1) % len(phases)]
            next_position = self.node_position[
                (connected.connection.bus, next_phase)
            ]
            return (
                abs(voltage[position] - voltage[next_position])
                / (
                    math.sqrt(3.0)
                    * self.case.buses[connected.connection.bus].kv_base_ln
                )
            )
        return (
            abs(voltage[position])
            / self.case.buses[connected.connection.bus].kv_base_ln
        )

    def _network_margins(self, evaluated):
        margins = []
        for voltage in evaluated["voltages"]:
            for position, (bus_id, phase) in enumerate(self.nodes):
                magnitude = abs(voltage[position]) / self.voltage_bases_kv[position]
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
            voltage[phase] = pd.Series(
                abs(values) / model.voltage_bases_kv[position], index=index
            )
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
            sending_voltage = (
                model.branch_from_connections[branch_index] @ from_voltage
            )
            receiving_voltage = (
                model.branch_to_connections[branch_index] @ to_voltage
            )
            sending = sending_voltage * np.conj(current) * 1000.0
            to_current = model.branch_taps[branch_index].conj().T @ current
            receiving = receiving_voltage * np.conj(to_current) * 1000.0
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

    inverter_losses_kw = 0.0
    for device_index, connected in enumerate(case.bess):
        connected.result = BessResult(
            p_net_kw={
                phase: pd.Series(
                    solution.p_charge_phase_kw[
                        device_index, model.phase_position[phase]
                    ]
                    - solution.p_discharge_phase_kw[
                        device_index, model.phase_position[phase]
                    ],
                    index=index,
                )
                for phase in connected.connection.phases
            },
            q_injection_kvar={
                phase: pd.Series(
                    solution.q_bess_phase_kvar[
                        device_index, model.phase_position[phase]
                    ],
                    index=index,
                )
                for phase in connected.connection.phases
            },
            soc_kwh=pd.Series(solution.soc_kwh[device_index], index=index),
            soc_frac=pd.Series(
                solution.soc_kwh[device_index] / connected.device.e_cap_kwh,
                index=index,
            ),
        )
        if connected.connection.dispatch_mode == "aggregate":
            inverter_losses_kw += sum(
                model._bess_q_loss(device_index, value)
                for value in solution.q_bess_kvar[device_index]
            )
        else:
            phase_indices = [
                model.phase_position[phase]
                for phase in connected.connection.phases
            ]
            inverter_losses_kw += sum(
                model._bess_q_loss(
                    device_index,
                    solution.q_bess_phase_kvar[
                        device_index, phase_indices, time
                    ],
                )
                for time in case.periods
            )

    pv_curtailed = 0.0
    for device_index, connected in enumerate(case.pv):
        generated = {}
        reactive = {}
        curtailed = {}
        for phase in connected.connection.phases:
            phase_index = model.phase_position[phase]
            available = connected.available_kw[phase].to_numpy()
            phase_generation = solution.pv_generation_phase_kw[
                device_index, phase_index
            ]
            generated[phase] = pd.Series(phase_generation, index=index)
            reactive[phase] = pd.Series(
                solution.q_pv_phase_kvar[device_index, phase_index],
                index=index,
            )
            curtailed[phase] = pd.Series(
                np.maximum(available - phase_generation, 0.0), index=index
            )
        connected.result = PvResult(
            available_kw=connected.available_kw,
            generation_kw=generated,
            q_injection_kvar=reactive,
            curtailment_kw=curtailed,
        )
        pv_curtailed += sum(float(values.sum()) for values in curtailed.values())
        if connected.connection.dispatch_mode == "aggregate":
            inverter_losses_kw += sum(
                model._pv_q_loss(device_index, value)
                for value in solution.q_pv_kvar[device_index]
            )
        else:
            phase_indices = [
                model.phase_position[phase]
                for phase in connected.connection.phases
            ]
            inverter_losses_kw += sum(
                model._pv_q_loss(
                    device_index,
                    solution.q_pv_phase_kvar[
                        device_index, phase_indices, time
                    ],
                )
                for time in case.periods
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
        pv_curtailed_kwh=pv_curtailed * case.dt_h,
        losses_kwh=total_losses_kw * case.dt_h,
        v_min_pu=min(all_voltages),
        v_max_pu=max(all_voltages),
        socp_gap_max_pu2=float("nan"),
        socp_gap_max_normalized=float("nan"),
        socp_gap_tolerance=float("nan"),
        socp_tightness_margin=float("nan"),
        socp_tightness="not_applicable",
        socp_relaxation_tight=False,
        inverter_losses_kwh=inverter_losses_kw * case.dt_h,
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
