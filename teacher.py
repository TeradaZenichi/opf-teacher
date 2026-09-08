"""OPF teacher for local and central state-action pairs."""
from __future__ import annotations

from copy import deepcopy
from collections import deque
import math
from pathlib import Path

import pyomo.environ as pyo

from opf.active_model import build_active_power_model, _attach_active_results
from opf.components import Case
from opf.data import load_case
from opf.model import build_model
from opf.results import DEFAULT_SOCP_GAP_TOLERANCE, attach_results
from opf.states import (BessAction, BessState, BusState, GridState, PvAction, PvState)


class Teacher:
    def __init__(self, source: str | Path | Case, active_power_only: bool = False):
        self.case: Case = source if isinstance(source, Case) else load_case(source)
        self.model: pyo.ConcreteModel | None = None
        self.active_power_only = active_power_only

    def build(self) -> "Teacher":
        self._clear_results()
        self.model = None
        builder = build_active_power_model if self.active_power_only else build_model
        self.model = builder(self.case)
        return self

    def observe(self, *, buses: dict[int, BusState], bess: dict[str, BessState], pv: dict[str, PvState]) -> "Teacher":
        """Observe the first interval; later case values remain forecasts.

        Voltage is a pre-action feature, not an OPF constraint. Supply a new
        Case to advance the horizon.
        """
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

    def solve(self, solver: str | None = None, tee: bool = False,
              socp_gap_tolerance: float = DEFAULT_SOCP_GAP_TOLERANCE,
              **options) -> Case:
        if self.model is None:
            self.build()
        self._clear_results()
        solver = solver or ("appsi_highs" if self.active_power_only else "gurobi_direct")
        optimizer = pyo.SolverFactory(solver)
        if optimizer is None or not optimizer.available(exception_flag=False):
            raise RuntimeError(f"Solver '{solver}' unavailable")
        for name, value in options.items():
            optimizer.options[name] = value
        result = optimizer.solve(self.model, tee=tee)
        status = str(result.solver.termination_condition)
        if status.lower() not in {"optimal", "locallyoptimal"}:
            raise RuntimeError(f"Teacher did not solve to optimality: {status}")
        if self.active_power_only:
            _attach_active_results(self.model, self.case, status)
        else:
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
