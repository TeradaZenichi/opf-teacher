"""OPF teacher for local and central state-action pairs."""
from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import pyomo.environ as pyo
import pandas as pd

from opf_teacher.normalization import PUNormalizer
from opf_teacher.opf.formulations import Formulation, parse_formulation
from opf_teacher.opf.common import OBSERVATION_SCHEMA_VERSION, resolve_case_source

from opf_teacher.opf import three_phase
from opf_teacher.opf.single_phase import components, data, states
from opf_teacher.opf.single_phase import active_model, distflow_socp, results
from opf_teacher.opf.single_phase.scaling import scale_model
from opf_teacher.opf.single_phase.conic import ConicModel
from opf_teacher.opf.single_phase.quality import analyze_quality


@dataclass
class SolveConfig:
    """Solver settings; solver=None selects the formulation's default."""

    solver: str | None = None
    tee: bool = False
    socp_gap_tolerance: float = results.DEFAULT_SOCP_GAP_TOLERANCE
    maxiter: int = 300
    ftol: float = 1e-8
    solver_options: dict = field(default_factory=dict)
    quality_tolerance: float = 1e-6


class Teacher:
    """Own a private case copy; observe/build/solve modify only that copy."""

    def __init__(
        self,
        source: str | Path | components.Case | three_phase.components.Case,
        formulation: str | Formulation | None = None,
    ):
        selected_formulation = formulation
        if (
            selected_formulation is None
            and not isinstance(source, (components.Case, three_phase.components.Case))
        ):
            _, config_path = resolve_case_source(source)
            with open(config_path, encoding="utf-8") as stream:
                selected_formulation = json.load(stream).get("formulation")
        if selected_formulation is None:
            selected_formulation = Formulation.SINGLE_PHASE_SOCP
        self.formulation = parse_formulation(selected_formulation)
        if isinstance(source, (components.Case, three_phase.components.Case)):
            self.case: components.Case | three_phase.components.Case = deepcopy(source)
        elif self.formulation is Formulation.THREE_PHASE_IVR:
            self.case = three_phase.data.load_case(source)
        else:
            self.case = data.load_case(source)
        if self.formulation is Formulation.THREE_PHASE_IVR:
            if not isinstance(self.case, three_phase.components.Case):
                raise TypeError("three_phase_ivr requires an opf_teacher.opf.three_phase.Case")
        elif not isinstance(self.case, components.Case):
            raise TypeError("Single-phase formulations require an opf_teacher.opf.single_phase.components.Case")
        self.normalizer = PUNormalizer(self.case.base.s_base_kva, self.case.base.v_base_kv)
        self.model: object | None = None

    def build(self, tie_break_penalty: float = active_model.DEFAULT_TIE_BREAK_PENALTY) -> "Teacher":
        self._clear_results()
        self.model = None
        if self.formulation is Formulation.SINGLE_PHASE_ACTIVE:
            self.model = active_model.ActivePowerModel(self.case, tie_break_penalty).build()
        elif self.formulation is Formulation.SINGLE_PHASE_SOCP:
            self.model = distflow_socp.DistFlowModel(self.case).build()
        else:
            self.model = three_phase.ivr_model.build_model(self.case)
        return self

    def observe(self, *, buses: dict[int, states.BusState | three_phase.states.BusState],
                bess: dict[str, states.BessState | three_phase.states.BessState],
                pv: dict[str, states.PvState | three_phase.states.PvState]) -> "Teacher":
        """Observe the first interval; later case values remain forecasts.

        Voltage is a pre-action feature, not an OPF constraint. Supply a new
        case to advance the horizon.
        """
        if self.formulation is Formulation.THREE_PHASE_IVR:
            return self._observe_three_phase(buses=buses, bess=bess, pv=pv)

        case = self.case
        self._validate_observations(buses, bess, pv)
        price = float(case.price.iloc[0])

        for bid, bus in case.buses.items():
            bus.state = deepcopy(buses[bid])
            bus.p_load_kw = bus.p_load_kw.astype(float)
            bus.q_load_kw = bus.q_load_kw.astype(float)
            bus.p_load_kw.iloc[0] = bus.state.p_load_kw
            bus.q_load_kw.iloc[0] = bus.state.q_load_kvar
        for device in case.bess:
            device.state = deepcopy(bess[device.id])
            # A new measured SoC must not move the terminal target.
            if device.cyclic_soc and device.soc_terminal_frac is None:
                device.soc_terminal_frac = device.soc_init_frac
            device.soc_init_frac = device.state.soc_before_frac
        for device in case.pv:
            device.state = deepcopy(pv[device.id])
            device.avail_kw = device.avail_kw.astype(float)
            device.avail_kw.iloc[0] = device.state.available_kw
        case.grid.state = states.GridState(price, price * case.grid.feed_in_ratio)
        self.model = None
        self._clear_results()
        return self

    def _observe_three_phase(self, *, buses, bess, pv):
        case = self.case
        self._validate_three_phase_observations(buses, bess, pv)
        price = float(case.price.iloc[0])

        for bus_id, bus in case.buses.items():
            bus.state = deepcopy(buses[bus_id])
            for phase in bus.phases:
                bus.p_load_kw[phase].iloc[0] = bus.state.p_load_kw[phase]
                bus.q_load_kvar[phase].iloc[0] = bus.state.q_load_kvar[phase]
        for connected in case.bess:
            connected.state = deepcopy(bess[connected.id])
            device = connected.device
            if device.cyclic_soc and device.soc_terminal_frac is None:
                device.soc_terminal_frac = device.soc_init_frac
            device.soc_init_frac = connected.state.soc_before_frac
        for connected in case.pv:
            connected.state = deepcopy(pv[connected.id])
            for phase in connected.connection.phases:
                connected.available_kw[phase].iloc[0] = (
                    connected.state.available_kw[phase]
                )
            connected.device.avail_kw.iloc[0] = sum(
                connected.state.available_kw.values()
            )
        case.grid.state = three_phase.states.GridState(
            price,
            price * case.grid.feed_in_ratio,
        )
        self.model = None
        self._clear_results()
        return self

    def observe_dict(self, observation: dict) -> "Teacher":
        """Consume named pu observations and convert only at the model boundary."""

        version = observation.get("observation_schema_version")
        if version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported observation_schema_version {version!r}; "
                f"expected {OBSERVATION_SCHEMA_VERSION}"
            )
        if observation.get("units") != "pu":
            raise ValueError("Named observations must use units='pu'")
        if observation.get("phase_power_base") != "total_network":
            raise ValueError("Phase powers must use the total network base")
        self.normalizer.check_base(observation.get("base", {}))
        self.normalizer.require_pu(observation)
        observation = self.normalizer.denormalize(observation)
        timestamp = pd.Timestamp(observation["timestamp"])
        if timestamp != self.case.index[0]:
            raise ValueError("Observation timestamp must match the first case timestamp")
        if not math.isclose(
            float(observation["dt_h"]), self.case.dt_h, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("Observation dt_h must match the case interval")
        self._validate_observed_grid(observation["grid"])

        bus_values = self._bus_observations_by_id(observation["buses"])
        buses = {}
        bess = {}
        pv = {}
        if self.formulation is Formulation.THREE_PHASE_IVR:
            for bus_id, values in bus_values.items():
                buses[bus_id] = three_phase.states.BusState(
                    values["v_before_pu"], values["angle_before_deg"],
                    values["p_load_kw"], values["q_load_kvar"],
                )
            for device_id, values in observation["bess"].items():
                bess[device_id] = three_phase.states.BessState(
                    values["soc_before_frac"], values["previous_p_kw"], values["previous_q_kvar"],
                )
            for device_id, values in observation["pv"].items():
                pv[device_id] = three_phase.states.PvState(
                    values["available_kw"], values["previous_p_kw"], values["previous_q_kvar"],
                )
        else:
            for bus_id, values in bus_values.items():
                buses[bus_id] = states.BusState(
                    self._single_phase(values["v_before_pu"]),
                    self._single_phase(values["p_load_kw"]),
                    self._single_phase(values["q_load_kvar"]),
                )
            for device_id, values in observation["bess"].items():
                bess[device_id] = states.BessState(
                    values["soc_before_frac"],
                    self._single_phase(values["previous_p_kw"]),
                    self._single_phase(values["previous_q_kvar"]),
                )
            for device_id, values in observation["pv"].items():
                pv[device_id] = states.PvState(
                    self._single_phase(values["available_kw"]),
                    self._single_phase(values["previous_p_kw"]),
                    self._single_phase(values["previous_q_kvar"]),
                )
        return self.observe(buses=buses, bess=bess, pv=pv)

    def _bus_observations_by_id(self, observations):
        name_to_id = {
            str(name).lower(): bus_id
            for name, bus_id in self.case.bus_name_to_id.items()
        }
        result = {}
        for key, values in observations.items():
            if key in self.case.buses:
                bus_id = key
            else:
                try:
                    bus_id = name_to_id[str(key).lower()]
                except KeyError as exc:
                    raise ValueError(f"Unknown observed bus {key!r}") from exc
            if bus_id in result:
                raise ValueError(f"Duplicate observation for bus {bus_id}")
            result[bus_id] = values
        return result

    def _validate_observed_grid(self, grid):
        expected_buy = float(self.case.price.iloc[0])
        expected_sell = expected_buy * self.case.grid.feed_in_ratio
        if not math.isclose(
            float(grid["buy_price_per_kwh"]), expected_buy,
            rel_tol=0.0, abs_tol=1e-12,
        ) or not math.isclose(
            float(grid["sell_price_per_kwh"]), expected_sell,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("Observed grid prices must match the case tariffs")

    @staticmethod
    def _single_phase(values):
        if set(values) != {"a"}:
            raise ValueError("Single-phase observations must contain only phase 'a'")
        return float(values["a"])

    def solve(self, solver: str | None = None, tee: bool = False,
              socp_gap_tolerance: float = results.DEFAULT_SOCP_GAP_TOLERANCE,
              *, config: SolveConfig | None = None, **legacy_options) -> components.Case | three_phase.components.Case:
        """Solve the owned case. Keyword options remain for API compatibility.

        New callers should put solver settings in SolveConfig.
        """
        if config is not None:
            if solver is not None or tee or legacy_options or socp_gap_tolerance != results.DEFAULT_SOCP_GAP_TOLERANCE:
                raise ValueError("Use config or legacy solver arguments, not both")
            settings = config
        else:
            settings = SolveConfig(solver, tee, socp_gap_tolerance)
            if self.formulation is Formulation.THREE_PHASE_IVR:
                settings.maxiter = legacy_options.pop("maxiter", settings.maxiter)
                settings.ftol = legacy_options.pop("ftol", settings.ftol)
                if legacy_options:
                    raise TypeError("Unknown three-phase solver options: " + ", ".join(legacy_options))
            else:
                settings.solver_options = legacy_options
        if self.model is None:
            self.build()
        self._clear_results()
        if self.formulation is Formulation.THREE_PHASE_IVR:
            self._solve_three_phase(settings)
            self._set_three_phase_actions()
        else:
            self._solve_single_phase(settings)
            self._set_single_phase_actions()
        return self.case

    def state_action(self):
        """Return central X/Y in pu; the owned Case retains physical model units."""
        state, action = self.case.state_action()
        return self.normalizer.normalize(state), self.normalizer.normalize(action)

    def _solve_single_phase(self, settings: SolveConfig):
        solver = settings.solver
        if solver is None:
            if self.formulation is Formulation.SINGLE_PHASE_ACTIVE:
                solver = "appsi_highs"
            elif not any(self.model.component_data_objects(pyo.SOSConstraint, active=True)):
                solver = "clarabel"
            else:
                solver = "gurobi_direct"
        solver_model = self.model
        if self.formulation is Formulation.SINGLE_PHASE_SOCP:
            solver_model = scale_model(self.model, self.case)
        if solver == "clarabel":
            solver_info = ConicModel(solver_model).solve(settings)
            status = solver_info["status"]
        else:
            optimizer = pyo.SolverFactory(solver)
            if optimizer is None or not optimizer.available(exception_flag=False):
                raise RuntimeError(f"Solver '{solver}' unavailable")
            for name, value in settings.solver_options.items():
                optimizer.options[name] = value
            solution = optimizer.solve(solver_model, tee=settings.tee, load_solutions=False)
            status = str(solution.solver.termination_condition)
            if status.lower() not in {"optimal", "locallyoptimal"}:
                message = getattr(solution.solver, "termination_message", "")
                raise RuntimeError(f"Teacher did not solve to optimality: {status}. {message}")
            solver_model.solutions.load_from(solution)
            solver_info = {"name": solver, "status": status}
        if solver_model is not self.model:
            pyo.TransformationFactory("core.scale_model").propagate_solution(solver_model, self.model)
        if self.formulation is Formulation.SINGLE_PHASE_ACTIVE:
            active_model._attach_active_results(self.model, self.case, status)
        else:
            results.attach_results(self.model, self.case, status, socp_gap_tolerance=settings.socp_gap_tolerance)
            self.case.quality = analyze_quality(self.model, self.case, settings.quality_tolerance)
            self.case.quality["solver"] = solver_info
            if not self.case.quality["feasible"]:
                raise RuntimeError(f"Teacher solution violates scaled constraints: {self.case.quality['worst_constraint']}")

    def _solve_three_phase(self, settings: SolveConfig):
        if settings.solver not in {None, "scipy_slsqp", "slsqp"}:
            raise ValueError("three_phase_ivr currently requires solver='scipy_slsqp'")
        if settings.solver_options:
            raise ValueError("Use maxiter and ftol for the three-phase solver")
        self.model.solve(tee=settings.tee, maxiter=settings.maxiter, ftol=settings.ftol)

    def _set_single_phase_actions(self):
        for device in self.case.bess:
            power = float(device.result.p_net_kw.iloc[0])
            reactive_power = float(device.result.q_kvar.iloc[0])
            device.action = states.BessAction(power, reactive_power)
        for device in self.case.pv:
            generation = float(device.result.gen_kw.iloc[0])
            reactive_power = float(device.result.q_kvar.iloc[0])
            device.action = states.PvAction(generation, reactive_power)

    def _set_three_phase_actions(self):
        for device in self.case.bess:
            power = {}
            reactive_power = {}
            for phase in device.connection.phases:
                power[phase] = float(device.result.p_net_kw[phase].iloc[0])
                reactive_power[phase] = float(device.result.q_injection_kvar[phase].iloc[0])
            device.action = three_phase.states.BessAction(power, reactive_power)
        for device in self.case.pv:
            generation = {}
            reactive_power = {}
            for phase in device.connection.phases:
                generation[phase] = float(device.result.generation_kw[phase].iloc[0])
                reactive_power[phase] = float(device.result.q_injection_kvar[phase].iloc[0])
            device.action = three_phase.states.PvAction(generation, reactive_power)

    def _validate_observations(self, buses, bess, pv):
        case = self.case
        if not case.n_periods or not math.isfinite(case.dt_h) or case.dt_h <= 0:
            raise ValueError("The observed case requires periods and positive dt_h")
        price = float(case.price.iloc[0])
        if not math.isfinite(price) or not math.isfinite(case.grid.feed_in_ratio):
            raise ValueError("Observed grid tariffs must be finite")
        for devices in (case.bess, case.pv):
            if len({d.id for d in devices}) != len(devices):
                raise ValueError("Device IDs must be unique within each device type")

        groups = (
            ("buses", buses, set(case.buses), states.BusState),
            ("bess", bess, {d.id for d in case.bess}, states.BessState),
            ("pv", pv, {d.id for d in case.pv}, states.PvState),
        )
        for label, observations, ids, state_type in groups:
            if set(observations) != ids:
                raise ValueError(f"{label} observations must match case IDs")
            for key, state in observations.items():
                if not isinstance(state, state_type):
                    raise TypeError(f"{label}[{key!r}] must be {state_type.__name__}")
                if not all(math.isfinite(value) for value in states.state_values(state).values()):
                    raise ValueError(f"Non-finite observation in {label}[{key!r}]")

        if any(state.v_before_pu <= 0 for state in buses.values()):
            raise ValueError("Observed voltage must be positive")
        for device in case.bess:
            soc = bess[device.id].soc_before_frac
            if not device.soc_min_frac <= soc <= device.soc_max_frac:
                raise ValueError(f"Observed SoC outside limits for {device.id!r}")
            target = device.soc_terminal_frac
            if target is None and device.cyclic_soc:
                target = device.soc_init_frac
            if target is not None and not device.soc_min_frac <= target <= device.soc_max_frac:
                raise ValueError(f"Terminal SoC outside limits for {device.id!r}")
        if any(state.available_kw < 0 for state in pv.values()):
            raise ValueError("Observed PV availability cannot be negative")

    def _validate_three_phase_observations(self, buses, bess, pv):
        case = self.case
        groups = (
            ("buses", buses, set(case.buses), three_phase.states.BusState),
            ("bess", bess, {device.id for device in case.bess}, three_phase.states.BessState),
            ("pv", pv, {device.id for device in case.pv}, three_phase.states.PvState),
        )
        for label, observations, ids, state_type in groups:
            if set(observations) != ids:
                raise ValueError(f"{label} observations must match case IDs")
            for key, state in observations.items():
                if not isinstance(state, state_type):
                    raise TypeError(f"{label}[{key!r}] must be {state_type.__name__}")

        for bus_id, bus in case.buses.items():
            if buses[bus_id].phases != bus.phases:
                raise ValueError(f"Bus {bus_id} observation phases do not match")
        for connected in case.bess:
            state = bess[connected.id]
            if state.phases != connected.connection.phases:
                raise ValueError(f"BESS {connected.id!r} observation phases do not match")
            device = connected.device
            if not device.soc_min_frac <= state.soc_before_frac <= device.soc_max_frac:
                raise ValueError(f"Observed SoC outside limits for {connected.id!r}")
        for connected in case.pv:
            if pv[connected.id].phases != connected.connection.phases:
                raise ValueError(f"PV {connected.id!r} observation phases do not match")

    def _clear_results(self):
        self.case.summary = None
        self.case.formulation = None
        self.case.quality = None
        for bus in self.case.buses.values():
            bus.result = None
        for branch in self.case.branches:
            branch.result = None
        self.case.grid.result = None
        for devices in (self.case.bess, self.case.pv):
            for device in devices:
                device.result = None
                device.action = None


class PairSelector:
    """Select central pairs or one BESS's local view of the global OPF."""

    def __init__(self, case, mode="central", device_id="b1"):
        if mode not in {"central", "local"}:
            raise ValueError("Observation mode must be 'central' or 'local'")
        self.mode = mode
        self.device_id = device_id if mode == "local" else None
        self.bus_id = None
        if mode == "local":
            devices = {device.id: device for device in case.bess}
            if device_id not in devices:
                raise ValueError(f"Unknown local BESS {device_id!r}")
            device = devices[device_id]
            if isinstance(case, components.Case):
                self.bus_id = device.bus
            else:
                self.bus_id = device.connection.bus

    def select(self, state, action):
        if self.mode == "central":
            return deepcopy(state), deepcopy(action)
        local_state = {
            "timestamp": state["timestamp"], "dt_h": state["dt_h"],
            "bus": state["buses"][self.bus_id],
            "device": state["bess"][self.device_id],
        }
        if "phase_order" in state:
            local_state["phase_order"] = state["phase_order"]
        return deepcopy(local_state), deepcopy(action["bess"][self.device_id])

    def as_dict(self):
        return {
            "mode": self.mode,
            "device_type": "bess" if self.mode == "local" else None,
            "device_id": self.device_id, "bus_id": self.bus_id,
        }


class StateActionHistory:
    """Collect observed states and pair them with the current action.

    A window of one returns the current state in a one-item list.
    Use one history per episode and controller.
    """

    def __init__(self, window: int = 1):
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer")
        self._history = deque(maxlen=window)

    @property
    def window(self):
        return self._history.maxlen

    def append(self, state: dict, action: dict) -> tuple[list[dict], dict] | None:
        """Append one observed step; return None until the window is full."""
        self._history.append(deepcopy(state))
        if len(self._history) < self.window:
            return None
        states = deepcopy(list(self._history))
        current_action = deepcopy(action)
        return states, current_action

    def reset(self):
        self._history.clear()


class TemporalTeacher(StateActionHistory):
    """Compatibility name retaining the previous default window of three."""

    def __init__(self, window=3):
        super().__init__(window)


class BessOpt(Teacher):
    """Backward-compatible name for Teacher."""


class ThreePhaseTeacher(Teacher):
    """Teacher for unbalanced three-phase cases."""

    def __init__(self, source):
        super().__init__(source, formulation=Formulation.THREE_PHASE_IVR)
