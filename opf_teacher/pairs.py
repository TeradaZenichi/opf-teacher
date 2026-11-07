"""Offline supervised pairs from single- and three-phase teacher trajectories."""
from copy import deepcopy
from dataclasses import dataclass, field
import gc
import math

from opf_teacher.opf import three_phase
from opf_teacher.opf.common import profile_value
from opf_teacher.opf.single_phase import states
from opf_teacher.opf.single_phase.power_flow import pre_action_buses
from opf_teacher.teacher import PairSelector, SolveConfig, StateActionHistory, Teacher


@dataclass
class PairConfig:
    formulation: str | None = None
    mode: str = "central"
    device_id: str = "b1"
    window: int | None = 1
    action_history: bool = False
    solve: SolveConfig = field(default_factory=SolveConfig)
    block_weeks: int | None = None
    pad_initial: bool = False
    extra_features: dict = field(default_factory=dict)


def _initial_states(case):
    is_three_phase = isinstance(case, three_phase.Case)
    state_types = three_phase.states if is_three_phase else states
    bess, pv = {}, {}
    for device in case.bess:
        parameters = device.device if is_three_phase else device
        zero = dict.fromkeys(device.connection.phases, 0.0) if is_three_phase else 0.0
        bess[device.id] = state_types.BessState(parameters.soc_init_frac, deepcopy(zero), deepcopy(zero))
    for device in case.pv:
        zero = dict.fromkeys(device.connection.phases, 0.0) if is_three_phase else 0.0
        pv[device.id] = state_types.PvState(deepcopy(zero), deepcopy(zero), deepcopy(zero))
    return bess, pv


def _observe(teacher, bess, pv, measurements=None):
    is_three_phase = isinstance(teacher.case, three_phase.Case)
    for device in teacher.case.pv:
        pv[device.id].available_kw = (
            {phase: float(values.iloc[0]) for phase, values in device.available_kw.items()}
            if is_three_phase else float(device.avail_kw.iloc[0])
        )
    if is_three_phase:
        if measurements is not None:
            raise ValueError("Branch extras currently require a single-phase teacher")
        network = three_phase.ivr_model.IVRModel(teacher.case.horizon(0, 1))
        buses = network.pre_action_buses(bess, pv)
    else:
        buses = pre_action_buses(teacher.case, bess, pv, measurements)
    teacher.observe(buses=buses, bess=bess, pv=pv)


def _slice(values, step):
    if isinstance(values, dict):
        return {phase: _slice(profile, step) for phase, profile in values.items()}
    return values.iloc[step:step + 1].copy()


def _replay(plan, config, bess, pv):
    """Replay one optimized plan using pre-action snapshots and exact planned SoC."""
    teacher = Teacher(plan.case.horizon(0, 1), formulation=plan.formulation)
    case = plan.solve(config=config.solve)
    if not (case.quality or {}).get("valid_for_imitation", True):
        raise ValueError("Teacher solution is not valid for imitation; check the solution quality")
    is_three_phase = isinstance(case, three_phase.Case)
    state_types = three_phase.states if is_three_phase else states
    selector = PairSelector(case, config.mode, config.device_id, config.extra_features)
    for step in case.periods:
        interval = teacher.case
        interval.timestamps = case.timestamps[step:step + 1]
        interval.price = _slice(case.price, step)
        for bus_id, bus in interval.buses.items():
            bus.p_load_kw = _slice(case.buses[bus_id].p_load_kw, step)
            if is_three_phase:
                bus.q_load_kvar = _slice(case.buses[bus_id].q_load_kvar, step)
            else:
                bus.q_load_kw = _slice(case.buses[bus_id].q_load_kw, step)
        for device, planned in zip(interval.pv, case.pv):
            if is_three_phase:
                device.available_kw = _slice(planned.available_kw, step)
                device.device.avail_kw = _slice(planned.device.avail_kw, step)
            else:
                device.avail_kw = _slice(planned.avail_kw, step)
        needs_angles = "angle_before_deg" in config.extra_features.get("bus", []) and not is_three_phase
        measurements = {} if config.extra_features.get("branches") or needs_angles else None
        _observe(teacher, bess, pv, measurements)
        teacher._set_actions(step, source=case)
        interval.summary = case.summary
        state, action = teacher.state_action()
        if measurements is not None:
            state["branches"] = measurements["branches"]
            if "angle_before_deg" in config.extra_features.get("bus", []):
                for bus_id, angle in measurements["angles"].items():
                    state["buses"][bus_id]["angle_before_deg"] = angle
        yield selector.select(state, action)
        for device, planned in zip(interval.bess, case.bess):
            bess[device.id] = state_types.BessState(profile_value(planned.result.soc_frac, step),
                                                  device.action.p_net_kw, device.action.q_injection_kvar)
        for device in interval.pv:
            pv[device.id].previous_p_kw = deepcopy(device.action.generation_kw)
            pv[device.id].previous_q_kvar = deepcopy(device.action.q_injection_kvar)


def _planned_pairs(source, config):
    weeks = config.block_weeks
    if weeks is not None and (isinstance(weeks, bool) or not isinstance(weeks, int) or weeks < 1):
        raise ValueError("block_weeks must be a positive integer or None")
    formulation = config.formulation or ("three_phase_ivr" if isinstance(source, three_phase.Case) else None)
    original = Teacher(source, formulation=formulation)
    case = original.case
    size = case.n_periods if weeks is None else round(weeks * 168 / case.dt_h)
    if weeks is not None and (size < 1 or not math.isclose(size * case.dt_h, weeks * 168)):
        raise ValueError("block_weeks must contain a whole number of case intervals")
    is_three_phase = isinstance(case, three_phase.Case)
    parameters = [device.device if is_three_phase else device for device in case.bess]
    targets = {device.id: device.soc_terminal_frac if device.soc_terminal_frac is not None
               else device.soc_init_frac if device.cyclic_soc else None for device in parameters}
    bess, pv = _initial_states(case)
    for start in range(0, case.n_periods, size):
        periods = min(size, case.n_periods - start)
        plan = Teacher(case.horizon(start, periods), formulation=original.formulation)
        for connected in plan.case.bess:
            device = connected.device if is_three_phase else connected
            device.soc_init_frac = bess[device.id].soc_before_frac
            device.cyclic_soc = False
            device.soc_terminal_frac = targets[device.id] if start + periods == case.n_periods else None
        yield from _replay(plan, config, bess, pv)
        del plan
        gc.collect()


def collect_pairs(source, config=None):
    """Solve once per selected period or weekly block; keep one continuous history."""
    config = PairConfig() if config is None else config
    history = None if config.window is None else StateActionHistory(config.window, config.action_history, config.pad_initial)
    x, y = [], []
    for state, action in _planned_pairs(source, config):
        pair = (state, action) if history is None else history.append(state, action)
        if pair is not None:
            x.append(pair[0])
            y.append(pair[1])
    if not x:
        raise ValueError("No pairs; check the selected period and observation window")
    return x, y
