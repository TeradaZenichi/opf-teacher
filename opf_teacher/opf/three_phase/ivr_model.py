"""Unbalanced three-phase AC optimization."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.optimize import linprog, minimize

from opf_teacher.opf.common import (
    QMAX_FRAC,
    VV_Q,
    VV_V,
    VW_P,
    VW_V,
)
from opf_teacher.opf.three_phase.components import Case
from opf_teacher.opf.three_phase.results import attach_results


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
        return Dispatch(
            p_charge=self.p_charge.copy(),
            p_discharge=self.p_discharge.copy(),
            q_bess=self.q_bess.copy(),
            p_pv=self.p_pv.copy(),
            q_pv=self.q_pv.copy(),
            p_charge_phase=self.p_charge_phase.copy(),
            p_discharge_phase=self.p_discharge_phase.copy(),
            q_bess_phase=self.q_bess_phase.copy(),
            p_pv_phase=self.p_pv_phase.copy(),
            q_pv_phase=self.q_pv_phase.copy(),
        )


@dataclass
class DispatchEvaluation:
    """Physical trajectories and cost calculated for one candidate dispatch."""

    objective: float
    voltages: list[np.ndarray]
    currents: list[list[np.ndarray]]
    grid_power: list[np.ndarray]
    soc: np.ndarray
    dispatch: Dispatch
    residual: float


@dataclass
class DecisionVariables:
    """Order, bounds and initial values passed to SciPy."""

    layout: list = field(default_factory=list)
    bounds: list = field(default_factory=list)
    initial: list = field(default_factory=list)

    def add(self, name, indices, lower, upper, initial):
        self.layout.append((name, tuple(indices)))
        self.bounds.append((float(lower), float(upper)))
        self.initial.append(float(np.clip(initial, lower, upper)))


class IVRModel:
    """Three-phase AC-IVR problem."""

    def __init__(self, case: Case):
        if not isinstance(case, Case):
            raise TypeError("three_phase_ivr requires an opf_teacher.opf.three_phase.Case")
        self.case = case
        self.solution: IVRSolution | None = None
        self._last_dispatch_key: bytes | None = None
        self._last_evaluation: DispatchEvaluation | None = None
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
        supported_controls = {
            "optimal", "fixed_pf", "volt-var", "volt-watt", "volt-var-watt"
        }
        for connected in case.pv:
            if connected.device.control not in supported_controls:
                raise ValueError(
                    f"Unsupported three-phase PV control {connected.device.control!r}"
                )

        bounds, x0 = self._decision_bounds_and_start()

        self.solution = None
        self._last_dispatch_key = None
        self._last_evaluation = None

        constraints = []
        if self._has_device_equalities():
            constraints.append({
                "type": "eq",
                "fun": lambda values: self._device_equalities(values, self._evaluate_dispatch(values)),
            })
        if case.bess or case.pv:
            constraints.append({
                "type": "ineq",
                "fun": lambda values: self._device_margins(values, self._evaluate_dispatch(values)),
            })
        constraints.append({
            "type": "ineq",
            "fun": lambda values: self._network_margins(self._evaluate_dispatch(values)),
        })

        if x0.size:
            optimized = minimize(
                lambda values: self._evaluate_dispatch(values).objective,
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
            margins = self._network_margins(self._evaluate_dispatch(values))
            if margins.size and float(np.min(margins)) < -1e-7:
                raise RuntimeError("Fixed three-phase dispatch violates network limits")
            iterations = 0

        evaluated = self._evaluate_dispatch(values)
        dispatch = self._unpack_dispatch(values)
        self.solution = IVRSolution(
            voltages_kv=evaluated.voltages,
            branch_currents_ka=evaluated.currents,
            grid_power_kva=evaluated.grid_power,
            p_charge_kw=dispatch.p_charge,
            p_discharge_kw=dispatch.p_discharge,
            q_bess_kvar=dispatch.q_bess,
            p_charge_phase_kw=dispatch.p_charge_phase,
            p_discharge_phase_kw=dispatch.p_discharge_phase,
            q_bess_phase_kvar=dispatch.q_bess_phase,
            soc_kwh=evaluated.soc,
            pv_generation_kw=dispatch.p_pv,
            q_pv_kvar=dispatch.q_pv,
            pv_generation_phase_kw=dispatch.p_pv_phase,
            q_pv_phase_kvar=dispatch.q_pv_phase,
            objective_cost=evaluated.objective,
            iterations=iterations,
            max_power_flow_residual_a=evaluated.residual * 1000.0,
        )
        return attach_results(self, case)

    def _evaluate_dispatch(self, values) -> DispatchEvaluation:
        """Evaluate network physics and cost for one candidate dispatch."""
        case = self.case
        key = np.asarray(values, dtype=float).tobytes()
        if self._last_dispatch_key == key and self._last_evaluation is not None:
            return self._last_evaluation
        dispatch = self._unpack_dispatch(values)
        voltages = []
        branch_currents = []
        grid_power = []
        max_residual = 0.0
        previous_voltage = self.nominal_voltage
        for time in range(case.n_periods):
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
        value = DispatchEvaluation(
            objective=float(objective),
            voltages=voltages,
            currents=branch_currents,
            grid_power=grid_power,
            soc=soc,
            dispatch=dispatch,
            residual=max_residual,
        )
        self._last_dispatch_key = key
        self._last_evaluation = value
        return value


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
        variables = DecisionVariables()

        self._prepare_bess_variables(variables)
        self._prepare_pv_variables(variables, dispatch)

        self._fixed_dispatch = dispatch
        self._variable_layout = variables.layout
        self._variable_keys = set(variables.layout)
        return variables.bounds, np.asarray(variables.initial, dtype=float)

    def _prepare_bess_variables(self, variables: DecisionVariables):
        """Add battery power bounds and an economic starting dispatch."""
        periods = self.case.n_periods
        for index, connected in enumerate(self.case.bess):
            battery = connected.device
            device_bounds = (
                [(0.0, battery.p_charge_max_kw)] * periods
                + [(0.0, battery.p_discharge_max_kw)] * periods
            )
            initial = self._economic_initial_dispatch(connected, device_bounds)
            if connected.connection.dispatch_mode == "aggregate":
                for time in range(periods):
                    variables.add(
                        "p_charge", (index, time), 0.0,
                        battery.p_charge_max_kw, initial[time],
                    )
                for time in range(periods):
                    variables.add(
                        "p_discharge", (index, time), 0.0,
                        battery.p_discharge_max_kw, initial[periods + time],
                    )
                if battery.reactive_control:
                    for time in range(periods):
                        variables.add(
                            "q_bess", (index, time), -battery.s_max_kva,
                            battery.s_max_kva, 0.0,
                        )
            else:
                count = len(connected.connection.phases)
                for phase in connected.connection.phases:
                    phase_index = self.phase_position[phase]
                    for time in range(periods):
                        variables.add(
                            "p_charge_phase", (index, phase_index, time),
                            0.0, battery.p_charge_max_kw,
                            initial[time] / count,
                        )
                    for time in range(periods):
                        variables.add(
                            "p_discharge_phase", (index, phase_index, time),
                            0.0, battery.p_discharge_max_kw,
                            initial[periods + time] / count,
                        )
                    if battery.reactive_control:
                        for time in range(periods):
                            variables.add(
                                "q_bess_phase", (index, phase_index, time),
                                -battery.s_max_kva / count,
                                battery.s_max_kva / count,
                                0.0,
                            )

    def _prepare_pv_variables(self, variables: DecisionVariables, dispatch: Dispatch):
        """Set solar availability, curve initial values and power bounds."""
        for index, connected in enumerate(self.case.pv):
            pv = connected.device
            available = np.minimum(
                connected.available_total_kw.to_numpy(), pv.p_max_kw
            )
            dispatch.p_pv[index] = available
            if "var" in pv.control:
                initial_voltage = float(np.mean([
                    self.case.grid.v_ref_pu[phase]
                    for phase in connected.connection.phases
                ]))
                dispatch.q_pv[index] = QMAX_FRAC * pv.s_max_kva * np.interp(
                    initial_voltage,
                    VV_V,
                    VV_Q,
                )
            if "watt" in pv.control:
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
                    pv.curtailable
                    or pv.q_loss_rated_kw > 0.0
                    or "watt" in pv.control
                )
                if p_is_variable and connected.connection.dispatch_mode == "aggregate":
                    variables.add(
                        "p_pv", (index, time), 0.0, value,
                        dispatch.p_pv[index, time],
                    )
            for time, value in enumerate(available):
                q_enabled = (
                    pv.control in {"optimal", "volt-var", "volt-var-watt"}
                    and (value > 0.0 or pv.night_var)
                )
                if q_enabled and connected.connection.dispatch_mode == "aggregate":
                    variables.add(
                        "q_pv", (index, time), -pv.s_max_kva,
                        pv.s_max_kva, dispatch.q_pv[index, time],
                    )

            if connected.connection.dispatch_mode == "per_phase":
                count = len(connected.connection.phases)
                for phase in connected.connection.phases:
                    phase_index = self.phase_position[phase]
                    phase_available = np.minimum(
                        connected.available_kw[phase].to_numpy(),
                        pv.p_max_kw / count,
                    )
                    dispatch.p_pv_phase[index, phase_index] = phase_available
                    for time, value in enumerate(phase_available):
                        p_is_variable = (
                            pv.curtailable
                            or pv.q_loss_rated_kw > 0.0
                            or "watt" in pv.control
                        )
                        if p_is_variable:
                            variables.add(
                                "p_pv_phase", (index, phase_index, time),
                                0.0, value, value,
                            )
                        q_enabled = (
                            pv.control in {
                                "optimal", "volt-var", "volt-var-watt"
                            }
                            and (value > 0.0 or pv.night_var)
                        )
                        if q_enabled:
                            variables.add(
                                "q_pv_phase", (index, phase_index, time),
                                -pv.s_max_kva / count,
                                pv.s_max_kva / count,
                                0.0,
                            )

    def _unpack_dispatch(self, values):
        dispatch = self._fixed_dispatch.copy()
        arrays = {
            "p_charge": dispatch.p_charge,
            "p_discharge": dispatch.p_discharge,
            "q_bess": dispatch.q_bess,
            "p_pv": dispatch.p_pv,
            "q_pv": dispatch.q_pv,
            "p_charge_phase": dispatch.p_charge_phase,
            "p_discharge_phase": dispatch.p_discharge_phase,
            "q_bess_phase": dispatch.q_bess_phase,
            "p_pv_phase": dispatch.p_pv_phase,
            "q_pv_phase": dispatch.q_pv_phase,
        }
        for value, (name, indices) in zip(
            np.asarray(values, dtype=float), self._variable_layout
        ):
            arrays[name][indices] = value
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
        dispatch = evaluated.dispatch
        equalities = []
        for index, connected in enumerate(self.case.bess):
            device = connected.device
            if device.cyclic_soc or device.soc_terminal_frac is not None:
                target = (
                    device.soc_init_frac
                    if device.soc_terminal_frac is None
                    else device.soc_terminal_frac
                ) * device.e_cap_kwh
                equalities.append(evaluated.soc[index, -1] - target)
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
                            connected, evaluated.voltages[time]
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
                                connected, phase, evaluated.voltages[time]
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
        dispatch = evaluated.dispatch
        margins = []
        for index, connected in enumerate(self.case.bess):
            device = connected.device
            soc = evaluated.soc[index]
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
                                evaluated.voltages[time],
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
                        connected, evaluated.voltages[time]
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
        for voltage in evaluated.voltages:
            for position, (bus_id, phase) in enumerate(self.nodes):
                magnitude = abs(voltage[position]) / self.voltage_bases_kv[position]
                bus = self.case.buses[bus_id]
                margins.extend((
                    magnitude - bus.v_min_pu[phase],
                    bus.v_max_pu[phase] - magnitude,
                ))
        for currents_at_time in evaluated.currents:
            for branch, current in zip(self.case.branches, currents_at_time):
                for position, phase in enumerate(branch.phases):
                    margins.append(branch.norm_amps[phase] - abs(current[position]) * 1000.0)
        for phase_power in evaluated.grid_power:
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
