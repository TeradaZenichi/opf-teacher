from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from opf_teacher import PairConfig, StateActionHistory, collect_pairs, rollout
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


class PairsTest(unittest.TestCase):
    def test_eight_phase_scope_window_combinations(self):
        for formulation, case in cases():
            reference = {}
            for mode, window in product(("central", "local"), (1, 2)):
                with self.subTest(formulation=formulation, mode=mode, window=window):
                    config = PairConfig(formulation=formulation, mode=mode, window=window, horizon=2)
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

    def test_trajectory_carries_soc_and_powers_to_the_next_pre_action_state(self):
        for formulation, case in cases():
            with self.subTest(formulation=formulation):
                steps = list(rollout(case, PairConfig(formulation=formulation, horizon=2)))
                first, second = steps[0].case.bess[0], steps[1].case.bess[0]
                if second.state.soc_before_frac != float(first.result.soc_frac.iloc[0]):
                    self.fail("Only the first interval's SoC must initialize the next horizon")
                if second.state.previous_p_kw != first.action.p_net_kw:
                    self.fail("Previous actions must match the preceding interval's label")
                if formulation == "three_phase_ivr":
                    bus = case.bess[0].connection.bus
                    if steps[0].case.buses[bus].state.v_before_pu["a"] == 1.0:
                        self.fail("Initial voltage must be measured with the current demand")

    def test_action_windows_align_with_states_for_both_formulations(self):
        for formulation, case in cases():
            with self.subTest(formulation=formulation):
                config = PairConfig(formulation=formulation, mode="local", window=2, horizon=2)
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

    def test_invalid_action_labels_are_filtered_over_the_whole_target_window(self):
        case = cases()[0][1]
        steps = []
        for step, valid in enumerate((False, True, True)):
            solved = deepcopy(case)
            solved.quality = {"valid_for_imitation": valid}
            steps.append(SimpleNamespace(case=solved, state_action=lambda step=step: ({"step": step}, {"p": step})))
        with patch("opf_teacher.pairs.rollout", return_value=iter(steps)):
            x, y = collect_pairs(case, PairConfig(window=2, action_history=True))
        if x != [[{"step": 1}, {"step": 2}]] or y != [[{"p": 1}, {"p": 2}]]:
            self.fail("A sequence target must contain only valid teacher labels")


if __name__ == "__main__":
    unittest.main()
