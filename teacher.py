"""OPF teacher for local and central state-action pairs."""
from __future__ import annotations

from copy import deepcopy
from collections import deque
import json
import math
from pathlib import Path

import pyomo.environ as pyo
import pandas as pd

from opf.formulations import Formulation, resolve_formulation
from opf.case_source import resolve_case_source
from opf.contracts import OBSERVATION_SCHEMA_VERSION
from opf.single_phase.active_model import build_active_power_model, _attach_active_results
from opf.components import Case
from opf.data import load_case
from opf.single_phase.distflow_socp import build_model as build_single_phase_socp_model
from opf.single_phase.results import DEFAULT_SOCP_GAP_TOLERANCE, attach_results
from opf.states import (BessAction, BessState, BusState, GridState, PvAction, PvState)
from opf.three_phase.components import Case as ThreePhaseCase
from opf.three_phase.data import load_case as load_three_phase_case
from opf.three_phase.ivr_model import build_model as build_three_phase_ivr_model
from opf.three_phase.states import (
    BessAction as ThreePhaseBessAction,
    BessState as ThreePhaseBessState,
    BusState as ThreePhaseBusState,
    GridState as ThreePhaseGridState,
    PvAction as ThreePhasePvAction,
    PvState as ThreePhasePvState,
)


_BUILDERS = {
    Formulation.SINGLE_PHASE_ACTIVE: build_active_power_model,
    Formulation.SINGLE_PHASE_SOCP: build_single_phase_socp_model,
    Formulation.THREE_PHASE_IVR: build_three_phase_ivr_model,
}

_DEFAULT_SOLVERS = {
    Formulation.SINGLE_PHASE_ACTIVE: "appsi_highs",
    Formulation.SINGLE_PHASE_SOCP: "gurobi_direct",
    Formulation.THREE_PHASE_IVR: "scipy_slsqp",
}


class Teacher:
    def __init__(
        self,
        source: str | Path | Case | ThreePhaseCase,
        active_power_only: bool = False,
        formulation: str | Formulation | None = None,
    ):
        selected_formulation = formulation
        if (
            selected_formulation is None
            and not active_power_only
            and not isinstance(source, (Case, ThreePhaseCase))
        ):
            _, config_path = resolve_case_source(source)
            with open(config_path, encoding="utf-8") as stream:
                selected_formulation = json.load(stream).get("formulation")
        self.formulation = resolve_formulation(
            selected_formulation,
            active_power_only=active_power_only,
        )
        if isinstance(source, (Case, ThreePhaseCase)):
            self.case: Case | ThreePhaseCase = source
        elif self.formulation is Formulation.THREE_PHASE_IVR:
            self.case = load_three_phase_case(source)
        else:
            self.case = load_case(source)
        self.model: object | None = None
        # Kept for callers that still inspect the legacy flag.
        self.active_power_only = self.formulation is Formulation.SINGLE_PHASE_ACTIVE

    def build(self) -> "Teacher":
        self._clear_results()
        self.model = None
        builder = _BUILDERS[self.formulation]
        self.model = builder(self.case)
        return self

    def observe(self, *, buses: dict[int, BusState], bess: dict[str, BessState], pv: dict[str, PvState]) -> "Teacher":
        """Observe the first interval; later case values remain forecasts.

        Voltage is a pre-action feature, not an OPF constraint. Supply a new
        Case to advance the horizon.
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
        case.grid.state = GridState(price, price * case.grid.feed_in_ratio)
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
        case.grid.state = ThreePhaseGridState(
            price,
            price * case.grid.feed_in_ratio,
        )
        self.model = None
        self._clear_results()
        return self

    def observe_dict(self, observation: dict) -> "Teacher":
        """Consume the versioned named observation emitted by the environment."""

        version = observation.get("observation_schema_version")
        if version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported observation_schema_version {version!r}; "
                f"expected {OBSERVATION_SCHEMA_VERSION}"
            )
        timestamp = pd.Timestamp(observation["timestamp"])
        if timestamp != self.case.index[0]:
            raise ValueError("Observation timestamp must match the first case timestamp")
        if not math.isclose(
            float(observation["dt_h"]), self.case.dt_h, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("Observation dt_h must match the case interval")
        self._validate_observed_grid(observation["grid"])

        bus_values = self._bus_observations_by_id(observation["buses"])
        if self.formulation is Formulation.THREE_PHASE_IVR:
            buses = {
                bus_id: ThreePhaseBusState(**values)
                for bus_id, values in bus_values.items()
            }
            bess = {
                device_id: ThreePhaseBessState(**values)
                for device_id, values in observation["bess"].items()
            }
            pv = {
                device_id: ThreePhasePvState(**values)
                for device_id, values in observation["pv"].items()
            }
        else:
            buses = {
                bus_id: BusState(
                    self._single_phase(values["v_before_pu"]),
                    self._single_phase(values["p_load_kw"]),
                    self._single_phase(values["q_load_kvar"]),
                )
                for bus_id, values in bus_values.items()
            }
            bess = {
                device_id: BessState(
                    values["soc_before_frac"],
                    self._single_phase(values["previous_p_kw"]),
                    self._single_phase(values["previous_q_kvar"]),
                )
                for device_id, values in observation["bess"].items()
            }
            pv = {
                device_id: PvState(
                    self._single_phase(values["available_kw"]),
                    self._single_phase(values["previous_p_kw"]),
                    self._single_phase(values["previous_q_kvar"]),
                )
                for device_id, values in observation["pv"].items()
            }
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
              socp_gap_tolerance: float = DEFAULT_SOCP_GAP_TOLERANCE,
              **options) -> Case:
        if self.model is None:
            self.build()
        self._clear_results()
        solver = solver or _DEFAULT_SOLVERS[self.formulation]
        if self.formulation is Formulation.THREE_PHASE_IVR:
            if solver not in {"scipy_slsqp", "slsqp"}:
                raise ValueError("three_phase_ivr currently requires solver='scipy_slsqp'")
            solved = self.model.solve(tee=tee, **options)
            for device in solved.bess:
                device.action = ThreePhaseBessAction(
                    p_net_kw={
                        phase: float(values.iloc[0])
                        for phase, values in device.result.p_net_kw.items()
                    },
                    q_injection_kvar={
                        phase: float(values.iloc[0])
                        for phase, values in device.result.q_injection_kvar.items()
                    },
                )
            for device in solved.pv:
                device.action = ThreePhasePvAction(
                    generation_kw={
                        phase: float(values.iloc[0])
                        for phase, values in device.result.generation_kw.items()
                    },
                    q_injection_kvar={
                        phase: float(values.iloc[0])
                        for phase, values in device.result.q_injection_kvar.items()
                    },
                )
            return solved
        optimizer = pyo.SolverFactory(solver)
        if optimizer is None or not optimizer.available(exception_flag=False):
            raise RuntimeError(f"Solver '{solver}' unavailable")
        for name, value in options.items():
            optimizer.options[name] = value
        result = optimizer.solve(self.model, tee=tee)
        status = str(result.solver.termination_condition)
        if status.lower() not in {"optimal", "locallyoptimal"}:
            raise RuntimeError(f"Teacher did not solve to optimality: {status}")
        if self.formulation is Formulation.SINGLE_PHASE_ACTIVE:
            _attach_active_results(self.model, self.case, status)
        elif self.formulation is Formulation.SINGLE_PHASE_SOCP:
            attach_results(self.model, self.case, status, socp_gap_tolerance=socp_gap_tolerance)
        for device in self.case.bess:
            device.action = BessAction(float(device.result.p_net_kw.iloc[0]), float(device.result.q_kvar.iloc[0]))
        for device in self.case.pv:
            device.action = PvAction(float(device.result.gen_kw.iloc[0]), float(device.result.q_kvar.iloc[0]))
        return self.case

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
            ("buses", buses, set(case.buses), BusState),
            ("bess", bess, {d.id for d in case.bess}, BessState),
            ("pv", pv, {d.id for d in case.pv}, PvState),
        )
        for label, observations, ids, state_type in groups:
            if set(observations) != ids:
                raise ValueError(f"{label} observations must match case IDs")
            for key, state in observations.items():
                if not isinstance(state, state_type):
                    raise TypeError(f"{label}[{key!r}] must be {state_type.__name__}")
                if not all(math.isfinite(value) for value in vars(state).values()):
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
            ("buses", buses, set(case.buses), ThreePhaseBusState),
            ("bess", bess, {device.id for device in case.bess}, ThreePhaseBessState),
            ("pv", pv, {device.id for device in case.pv}, ThreePhasePvState),
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
        if hasattr(self.case, "formulation"):
            del self.case.formulation
        for item in [*self.case.buses.values(), *self.case.branches, self.case.grid, *self.case.bess, *self.case.pv]:
            item.result = None
        for device in [*self.case.bess, *self.case.pv]:
            device.action = None


class TemporalTeacher:
    """Pair a window of past/current observations with the current action."""

    def __init__(self, window=3):
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer")
        self._history = deque(maxlen=window)

    @property
    def window(self):
        return self._history.maxlen

    def append(self, x, y):
        """Append one observed step; return None until the window is full."""
        self._history.append(deepcopy(x))
        if len(self._history) < self.window:
            return None
        return deepcopy(list(self._history)), deepcopy(y)

    def reset(self):
        self._history.clear()


class BessOpt(Teacher):
    """Backward-compatible name for Teacher."""


class ThreePhaseTeacher(Teacher):
    """Phase-native teacher with an explicit formulation boundary."""

    def __init__(self, source):
        super().__init__(source, formulation=Formulation.THREE_PHASE_IVR)
