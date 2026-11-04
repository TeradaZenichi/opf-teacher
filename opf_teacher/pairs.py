"""Offline supervised pairs from single- and three-phase teacher trajectories."""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

from opf_teacher.opf import three_phase
from opf_teacher.opf.single_phase import states
from opf_teacher.opf.single_phase.power_flow import pre_action_buses
from opf_teacher.teacher import PairSelector, SolveConfig, StateActionHistory, Teacher


@dataclass
class PairConfig:
    formulation: str | None = None
    mode: str = "central"
    device_id: str = "b1"
    window: int = 1
    horizon: int | None = None
    action_history: bool = False
    solve: SolveConfig = field(default_factory=SolveConfig)


def rollout(source, config=None):
    """Yield solved horizons; advance only their first actions and predicted SoC."""
    config = PairConfig() if config is None else config
    formulation = config.formulation or ("three_phase_ivr" if isinstance(source, three_phase.Case) else None)
    original = Teacher(source, formulation=formulation)
    case = original.case
    is_three_phase = isinstance(case, three_phase.Case)
    state_types = three_phase.states if is_three_phase else states
    if config.horizon is not None and (isinstance(config.horizon, bool)
                                      or not isinstance(config.horizon, int) or config.horizon < 1):
        raise ValueError("horizon must be a positive integer or None")
    bess, pv = {}, {}
    for device in case.bess:
        parameters = device.device if is_three_phase else device
        zero = dict.fromkeys(device.connection.phases, 0.0) if is_three_phase else 0.0
        bess[device.id] = state_types.BessState(parameters.soc_init_frac, deepcopy(zero), deepcopy(zero))
    for device in case.pv:
        zero = dict.fromkeys(device.connection.phases, 0.0) if is_three_phase else 0.0
        pv[device.id] = state_types.PvState(deepcopy(zero), deepcopy(zero), deepcopy(zero))
    for step in case.periods:
        periods = None if config.horizon is None else min(config.horizon, case.n_periods - step)
        teacher = Teacher(case.horizon(step, periods), formulation=original.formulation)
        for device in teacher.case.pv:
            pv[device.id].available_kw = (
                {phase: float(values.iloc[0]) for phase, values in device.available_kw.items()}
                if is_three_phase else float(device.avail_kw.iloc[0])
            )
        if is_three_phase:
            network = three_phase.ivr_model.IVRModel(teacher.case.horizon(0, 1))
            buses = network.pre_action_buses(bess, pv)
        else:
            buses = pre_action_buses(teacher.case, bess, pv)
        teacher.observe(buses=buses, bess=bess, pv=pv)
        solved = teacher.solve(config=config.solve)
        yield teacher
        for device in solved.bess:
            bess[device.id] = state_types.BessState(float(device.result.soc_frac.iloc[0]),
                                                  device.action.p_net_kw, device.action.q_injection_kvar)
        for device in solved.pv:
            pv[device.id].previous_p_kw = deepcopy(device.action.generation_kw)
            pv[device.id].previous_q_kvar = deepcopy(device.action.q_injection_kvar)


def collect_pairs(source, config=None):
    """Return named pu X/Y lists; window=1 keeps a one-state X for compatibility."""
    config = PairConfig() if config is None else config
    history = StateActionHistory(config.window, action_history=config.action_history)
    valid_history = deque(maxlen=config.window)
    selector = None
    x, y = [], []
    for teacher in rollout(source, config):
        if selector is None:
            selector = PairSelector(teacher.case, config.mode, config.device_id)
        state, action = selector.select(*teacher.state_action())
        pair = history.append(state, action)
        valid = (teacher.case.quality or {}).get("valid_for_imitation", True)
        valid_history.append(valid)
        if pair is not None and (all(valid_history) if config.action_history else valid):
            x.append(pair[0])
            y.append(pair[1])
    if not x:
        raise ValueError("No valid pairs; check the period, window and teacher quality")
    return x, y
