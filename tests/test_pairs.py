from itertools import product
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from opf_teacher import PairConfig, StateActionHistory, Teacher, collect_pairs
from opf_teacher.opf import three_phase
from opf_teacher.opf.common import Bess, Pv
from opf_teacher.opf.single_phase import load_case


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def cases():
    single = load_case(FIXTURES / "single_phase_case").horizon(0, 2)
    single.bess[0].cyclic_soc = True
    single.grid.feed_in_ratio = 1.0
    triple = three_phase.load_case(FIXTURES / "three_phase_transformer_case")
    bus = triple.bus_name_to_id["load"]
    connection = three_phase.DeviceConnection(bus, ("a", "b", "c"))
    battery = Bess("b1", bus, 100.0, 2.0, 2.0, 0.95, 0.95, 0.5, 0.1, 0.9, False)
    triple.bess = [three_phase.ConnectedBess(battery, connection)]
    profile = pd.Series([0.9, 0.6], index=triple.index)
    pv = Pv("pv1", bus, 2.0, 2.0, "fixed_pf", False, 1.0, profile)
    triple.pv = [three_phase.ConnectedPv(pv, connection, {phase: profile / 3 for phase in connection.phases})]
    return (("single_phase_socp", single), ("three_phase_ivr", triple))


def retime(case, dt_h):
    case.dt_h = dt_h
    case.timestamps = list(pd.date_range(case.index[0], periods=case.n_periods, freq=pd.Timedelta(hours=dt_h)))
    profiles = [case.price]
    for bus in case.buses.values():
        profiles.extend(list(bus.p_load_kw.values()) + list(bus.q_load_kvar.values())
                        if isinstance(case, three_phase.Case) else [bus.p_load_kw, bus.q_load_kw])
    for device in case.pv:
        profiles.extend([device.device.avail_kw, *device.available_kw.values()]
                        if isinstance(case, three_phase.Case) else [device.avail_kw])
    for profile in profiles:
        profile.index = case.index
    return case


class PairsTest(unittest.TestCase):
    def test_initial_padding_matches_policy_windows_for_both_scopes_and_formulations(self):
        for formulation, case in cases():
            for mode in ("central", "local"):
                with self.subTest(formulation=formulation, mode=mode):
                    states, actions = collect_pairs(case, PairConfig(formulation=formulation, mode=mode, window=None))
                    config = PairConfig(formulation=formulation, mode=mode, window=3, pad_initial=True)
                    x, y = collect_pairs(case, config)
                    if x != [[states[0]] * 3, [states[0], states[0], states[1]]] or y != actions:
                        self.fail("Padded collection must match the policy's repeat-first initialization")
                    config.action_history = True
                    _, y = collect_pairs(case, config)
                    if y != [[None, None, actions[0]], [None, actions[0], actions[1]]]:
                        self.fail("Padded action histories must mark missing past labels")

    def test_initial_padding_is_not_restarted_at_weekly_boundaries(self):
        case = retime(cases()[0][1], 168)
        config = PairConfig(mode="local", window=2, block_weeks=1, pad_initial=True)
        x, _ = collect_pairs(case, config)
        if len(x) != 2 or x[0][0] != x[0][1] or x[1][0] != x[0][-1]:
            self.fail("Padding applies only at episode start, not at each optimization block")
        if x[1][0]["timestamp"] == x[1][1]["timestamp"]:
            self.fail("The next block must retain the real preceding observation")

    def test_weekly_blocks_carry_states_and_action_history_for_both_formulations(self):
        solve = Teacher.solve
        for formulation, case in cases():
            with self.subTest(formulation=formulation):
                retime(case, 168)
                parameters = case.bess[0].device if formulation == "three_phase_ivr" else case.bess[0]
                parameters.cyclic_soc = True
                target = parameters.soc_terminal_frac = 0.6
                solved, initial, targets = [], [], []

                def record(teacher, **kwargs):
                    device = teacher.case.bess[0]
                    device = device.device if formulation == "three_phase_ivr" else device
                    initial.append(device.soc_init_frac)
                    targets.append(device.soc_terminal_frac)
                    solved.append(solve(teacher, **kwargs))
                    return solved[-1]

                config = PairConfig(formulation=formulation, mode="local", window=2,
                                    action_history=True, block_weeks=1)
                with patch.object(Teacher, "solve", autospec=True, side_effect=record):
                    x, y = collect_pairs(case, config)
                if len(solved) != 2 or targets != [None, target] or len(x) != 1:
                    self.fail("Weekly blocks must preserve one history and enforce only the final target")
                previous_soc = float(solved[0].bess[0].result.soc_frac.iloc[-1])
                if initial[1] != previous_soc or x[0][1]["device"]["soc_before_frac"] != previous_soc:
                    self.fail("The next block must start from the preceding block's final SoC")
                if x[0][1]["device"]["previous_p_pu"] != y[0][0]["p_net_pu"]:
                    self.fail("Previous commands must cross block boundaries without resetting")
                if abs(float(solved[-1].bess[0].result.soc_frac.iloc[-1]) - target) > 1e-6:
                    self.fail("The final block must retain the configured terminal SoC")
                if parameters.soc_init_frac != 0.5 or case.summary is not None:
                    self.fail("Block collection must leave the caller's case unchanged")

    def test_weekly_blocks_keep_a_short_final_block_and_all_timestamps(self):
        case = retime(load_case(FIXTURES / "single_phase_case"), 56)
        case.grid.feed_in_ratio = 1.0
        case.bess[0].cyclic_soc = True
        solved = []
        solve = Teacher.solve

        def record(teacher, **kwargs):
            solved.append(solve(teacher, **kwargs))
            return solved[-1]

        with patch.object(Teacher, "solve", autospec=True, side_effect=record):
            x, y = collect_pairs(case, PairConfig(mode="local", window=None, block_weeks=1))
        if [block.n_periods for block in solved] != [3, 1] or len(y) != 4:
            self.fail("The last block must keep all remaining intervals")
        if [state["timestamp"] for state in x] != [time.isoformat() for time in case.index]:
            self.fail("Block collection must not duplicate or drop timestamps")
        windows, targets = collect_pairs(case, PairConfig(mode="local", window=3, block_weeks=1))
        if windows != [x[:3], x[1:]] or targets != y[2:]:
            self.fail("Observation windows must span consecutive optimization blocks")

    def test_block_weeks_validation_and_a_block_longer_than_the_case(self):
        case = cases()[0][1]
        for weeks in (0, -1, 1.5, True):
            with self.subTest(weeks=weeks), self.assertRaisesRegex(ValueError, "block_weeks"):
                collect_pairs(case, PairConfig(block_weeks=weeks))
        expected = collect_pairs(case)
        actual = collect_pairs(case, PairConfig(block_weeks=2))
        if actual != expected:
            self.fail("A block covering the entire case must retain single-optimization behavior")
        retime(case, 100)
        with self.assertRaisesRegex(ValueError, "whole number"):
            collect_pairs(case, PairConfig(block_weeks=1))

    def test_soc_roundoff_is_preserved_but_larger_violations_are_rejected(self):
        value = 1.0 + 1e-12
        state = three_phase.states.BessState(value, {"a": 0.0}, {"a": 0.0})
        if state.soc_before_frac != value:
            self.fail("Solver roundoff must be tolerated without changing the predicted SoC")
        with self.assertRaises(ValueError):
            three_phase.states.BessState(1.01, {"a": 0.0}, {"a": 0.0})

    def test_central_weekly_history_keeps_previous_pv_commands_and_phase_keys(self):
        case = retime(cases()[1][1], 168)
        config = PairConfig(formulation="three_phase_ivr", window=2, action_history=True, block_weeks=1)
        x, y = collect_pairs(case, config)
        pv = x[0][-1]["pv"]["pv1"]
        previous = y[0][0]["pv"]["pv1"]
        if pv["previous_p_pu"] != previous["generation_pu"] or pv["previous_q_pu"] != previous["q_injection_pu"]:
            self.fail("Previous PV commands must cross block boundaries with their phase keys")
        if set(pv["previous_p_pu"]) != {"a", "b", "c"} or "buses" not in x[0][-1]:
            self.fail("Central block trajectories must retain the full phase-aware observation")

    def test_single_solve_returns_aligned_local_and_central_trajectories(self):
        solve = Teacher.solve
        for formulation, case in cases():
            for mode in ("central", "local"):
                with self.subTest(formulation=formulation, mode=mode):
                    solved = []

                    def record(teacher, **kwargs):
                        solved.append(solve(teacher, **kwargs))
                        return solved[-1]

                    config = PairConfig(formulation=formulation, mode=mode, window=None)
                    with patch.object(Teacher, "solve", autospec=True, side_effect=record):
                        x, y = collect_pairs(case, config)
                    if len(solved) != 1 or len(x) != case.n_periods or len(y) != len(x):
                        self.fail("A flat trajectory must use one solve and retain every interval")
                    battery = solved[0].bess[0]
                    scale = case.base.s_base_kva
                    for step, (state, action) in enumerate(zip(x, y)):
                        local = state["device"] if mode == "local" else state["bess"]["b1"]
                        label = action if mode == "local" else action["bess"]["b1"]
                        expected = battery.result.p_net_kw
                        expected = ({p: float(v.iloc[step]) / scale for p, v in expected.items()}
                                    if isinstance(expected, dict) else float(expected.iloc[step]) / scale)
                        if label["p_net_pu"] != expected or state["timestamp"] != case.index[step].isoformat():
                            self.fail("Labels and timestamps must match the full optimized plan")
                        if step:
                            previous = y[step - 1] if mode == "local" else y[step - 1]["bess"]["b1"]
                            if (local["previous_p_pu"] != previous["p_net_pu"]
                                    or local["previous_q_pu"] != previous["q_injection_pu"]):
                                self.fail("Previous actions must generate the next pre-action observation")
                            if local["soc_before_frac"] != float(battery.result.soc_frac.iloc[step - 1]):
                                self.fail("Pre-action SoC must come from the preceding optimized interval")
                    if case.summary is not None or any(device.state is not None for device in case.bess):
                        self.fail("A single optimization must leave the caller's case unchanged")

    def test_single_solve_supports_current_targets_and_aligned_action_windows(self):
        for formulation, case in cases():
            for mode in ("central", "local"):
                with self.subTest(formulation=formulation, mode=mode):
                    config = PairConfig(formulation=formulation, mode=mode, window=None)
                    states, actions = collect_pairs(case, config)
                    for window in (1, 2):
                        config.window = window
                        config.action_history = False
                        x, y = collect_pairs(case, config)
                        expected = [states[i:i + window] for i in range(len(states) - window + 1)]
                        if x != expected or y != actions[window - 1:]:
                            self.fail("Sliding windows must retain the current-action contract")
                        config.action_history = True
                        x, y = collect_pairs(case, config)
                        if y != [actions[i:i + window] for i in range(len(actions) - window + 1)]:
                            self.fail("Action windows must align with every observed state")

    def test_flat_output_and_three_state_windows_use_one_optimization(self):
        case = load_case(FIXTURES / "single_phase_case")
        case.bess[0].cyclic_soc = True
        case.grid.feed_in_ratio = 1.0
        config = PairConfig(mode="local", window=None)
        solve = Teacher.solve
        with patch.object(Teacher, "solve", autospec=True, side_effect=solve) as optimizer:
            states, actions = collect_pairs(case, config)
        if optimizer.call_count != 1 or isinstance(states[0], list):
            self.fail("Flat output must use one optimization")
        config.window = 3
        x, y = collect_pairs(case, config)
        if x != [states[:3], states[1:4]] or y != [actions[2], actions[3]]:
            self.fail("Three-state windows must produce the requested overlapping pairs")

    def test_eight_phase_scope_window_combinations(self):
        for formulation, case in cases():
            reference = {}
            for mode, window in product(("central", "local"), (1, 2)):
                with self.subTest(formulation=formulation, mode=mode, window=window):
                    config = PairConfig(formulation=formulation, mode=mode, window=window)
                    x, y = collect_pairs(case, config)
                    if len(x) != 3 - window or len(x) != len(y) or any(len(row) != window for row in x):
                        self.fail("Unexpected pair count or observation window")
                    if mode == "central":
                        reference[window] = x, y
                    else:
                        central_x, central_y = reference[window]
                        bus_id = case.bess[0].connection.bus if formulation == "three_phase_ivr" else case.bess[0].bus
                        for local, central, target, full_target in zip(x, central_x, y, central_y):
                            if target != full_target["bess"]["b1"] or "buses" in local[-1]:
                                self.fail("Local labels must come from the global solution without remote observations")
                            if local[-1]["bus"] != central[-1]["buses"][bus_id]:
                                self.fail("Local observations must use the connected bus")
                    if formulation == "three_phase_ivr":
                        action = y[0]["bess"]["b1"] if mode == "central" else y[0]
                        if set(action["p_net_pu"]) != {"a", "b", "c"}:
                            self.fail("Three-phase actions must retain named phases")
            if case.summary is not None or any(device.state is not None for device in case.bess):
                self.fail("Collection must leave the caller's case unchanged")

    def test_initial_state_uses_current_demand_and_zero_previous_commands(self):
        for formulation, case in cases():
            with self.subTest(formulation=formulation):
                x, _ = collect_pairs(case, PairConfig(formulation=formulation, mode="local", window=None))
                voltage = x[0]["bus"]["v_before_pu"]
                previous = x[0]["device"]["previous_p_pu"]
                if any(value != 0.0 for value in (previous.values() if isinstance(previous, dict) else [previous])):
                    self.fail("The first observation must use zero previous device commands")
                if all(value == 1.0 for value in (voltage.values() if isinstance(voltage, dict) else [voltage])):
                    self.fail("Initial voltage must be measured with the current demand")

    def test_action_windows_align_with_states_for_both_formulations(self):
        for formulation, case in cases():
            with self.subTest(formulation=formulation):
                config = PairConfig(formulation=formulation, mode="local", window=2)
                current_x, current_y = collect_pairs(case, config)
                config.action_history = True
                x, y = collect_pairs(case, config)
                if x != current_x or y[0][-1] != current_y[0] or len(y[0]) != 2:
                    self.fail("Action-history mode must preserve the default current-action pair")
                if x[0][1]["device"]["previous_p_pu"] != y[0][0]["p_net_pu"]:
                    self.fail("The previous action must align with the next state's action history")

    def test_action_history_reset_and_independent_copies(self):
        history = StateActionHistory(2, action_history=True)
        action = {"p": [1]}
        history.append({"t": 0}, action)
        action["p"][0] = 99
        x, y = history.append({"t": 1}, {"p": [2]})
        if y != [{"p": [1]}, {"p": [2]}]:
            self.fail("Action windows must own independent copies")
        y[1]["p"][0] = 99
        if history.append({"t": 2}, {"p": [3]})[1][0] != {"p": [2]}:
            self.fail("Changing returned actions must not corrupt later windows")
        history.reset()
        if history.append({"t": 10}, {}) is not None:
            self.fail("Reset must clear both state and action history")

    def test_invalid_solution_is_rejected_before_generating_pairs(self):
        solve = Teacher.solve

        def invalid(teacher, **kwargs):
            case = solve(teacher, **kwargs)
            case.quality = {"valid_for_imitation": False}
            return case

        for formulation, case in cases():
            with self.subTest(formulation=formulation):
                config = PairConfig(formulation=formulation, window=2, action_history=True)
                with patch.object(Teacher, "solve", autospec=True, side_effect=invalid):
                    with self.assertRaisesRegex(ValueError, "not valid for imitation"):
                        collect_pairs(case, config)


if __name__ == "__main__":
    unittest.main()
