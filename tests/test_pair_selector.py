from copy import deepcopy
import json
from pathlib import Path
import unittest

from opf_teacher.normalization import PerUnitNormalizer, TimeEncoder
from opf_teacher.opf import three_phase
from opf_teacher.opf.single_phase import load_case
from opf_teacher.teacher import PairSelector, StateActionHistory


ROOT = Path(__file__).resolve().parents[1]


class PairSelectorTest(unittest.TestCase):
    def setUp(self):
        self.case = load_case(ROOT / "examples" / "case5_central")
        self.state = {
            "timestamp": "2026-01-01T06:00:00", "dt_h": 1.0,
            "buses": {
                2: {"v_before_pu": 1.01, "p_load_kw": 100.0, "q_load_kvar": 20.0},
                4: {"v_before_pu": 0.99, "p_load_kw": 70.0, "q_load_kvar": 15.0},
            },
            "grid": {"buy_price_per_kwh": 0.5, "sell_price_per_kwh": 0.25},
            "bess": {
                "b1": {"soc_before_frac": 0.4, "previous_p_kw": -10.0, "previous_q_kvar": 5.0},
                "b2": {"soc_before_frac": 0.7, "previous_p_kw": 20.0, "previous_q_kvar": -5.0},
            },
            "pv": {"pv1": {"available_kw": 100.0, "previous_p_kw": 90.0}},
        }
        self.action = {
            "bess": {
                "b1": {"p_net_kw": -20.0, "q_injection_kvar": 5.0},
                "b2": {"p_net_kw": 10.0, "q_injection_kvar": -5.0},
            },
            "pv": {"pv1": {"generation_kw": 80.0, "q_injection_kvar": 10.0}},
        }

    def test_central_selection_keeps_every_input_and_action(self):
        selector = PairSelector(self.case, "central", "unused")
        state, action = selector.select(self.state, self.action)
        self.assertEqual(state, self.state)
        self.assertEqual(action, self.action)
        state["bess"]["b1"]["soc_before_frac"] = 0.1
        action["bess"]["b1"]["p_net_kw"] = 999.0
        self.assertEqual(self.state["bess"]["b1"]["soc_before_frac"], 0.4)
        self.assertEqual(self.action["bess"]["b1"]["p_net_kw"], -20.0)
        self.assertEqual(selector.as_dict()["mode"], "central")
        self.assertIsNone(selector.as_dict()["device_id"])

    def test_local_selection_uses_the_selected_bess_bus_and_global_label(self):
        for device_id, bus_id in (("b1", 4), ("b2", 2)):
            with self.subTest(device=device_id):
                selector = PairSelector(self.case, "local", device_id)
                state, action = selector.select(self.state, self.action)
                self.assertEqual(set(state), {"timestamp", "dt_h", "bus", "device"})
                self.assertEqual(state["bus"], self.state["buses"][bus_id])
                self.assertEqual(state["device"], self.state["bess"][device_id])
                self.assertEqual(action, self.action["bess"][device_id])
                self.assertEqual(selector.as_dict()["bus_id"], bus_id)
                self.assertEqual(selector.as_dict()["device_type"], "bess")

    def test_local_pair_is_independent_of_the_global_pair(self):
        state, action = PairSelector(self.case, "local", "b1").select(self.state, self.action)
        state["bus"]["v_before_pu"] = 0.8
        state["device"]["soc_before_frac"] = 0.1
        action["p_net_kw"] = 999.0
        self.assertEqual(self.state["buses"][4]["v_before_pu"], 0.99)
        self.assertEqual(self.state["bess"]["b1"]["soc_before_frac"], 0.4)
        self.assertEqual(set(self.action["bess"]), {"b1", "b2"})
        self.assertEqual(self.action["bess"]["b1"]["p_net_kw"], -20.0)
        self.assertIn("pv1", self.action["pv"])

    def test_three_phase_local_view_keeps_phases_and_one_soc(self):
        case = three_phase.load_case(ROOT / "examples" / "case5_unbalanced")
        state = {
            "timestamp": self.state["timestamp"], "dt_h": 1.0,
            "phase_order": ["a", "b", "c"],
            "buses": {4: {
                "v_before_pu": {"a": 1.0, "b": 0.99, "c": 0.98},
                "angle_before_deg": {"a": -0.1, "b": -120.2, "c": 119.8},
                "p_load_kw": {"a": 10.0, "b": 20.0, "c": 30.0},
            }},
            "bess": {"b1": {
                "soc_before_frac": 0.4,
                "previous_p_kw": {"a": -1.0, "b": 0.0, "c": 1.0},
            }},
        }
        action = {"bess": {"b1": {"p_net_kw": {"a": -2.0, "b": 0.0, "c": 2.0}}}}
        selector = PairSelector(case, "local", "b1")
        local_state, local_action = selector.select(state, action)
        self.assertEqual(local_state["bus"], state["buses"][4])
        self.assertEqual(local_state["device"]["soc_before_frac"], 0.4)
        self.assertEqual(local_state["phase_order"], ["a", "b", "c"])
        self.assertEqual(local_action, action["bess"]["b1"])
        local_state["bus"]["v_before_pu"]["a"] = 0.8
        self.assertEqual(state["buses"][4]["v_before_pu"]["a"], 1.0)

    def test_local_pairs_keep_normalization_time_and_window_semantics(self):
        selector = PairSelector(self.case, "local", "b2")
        normalizer = PerUnitNormalizer(self.case.base)
        encoder = TimeEncoder()
        history = StateActionHistory(window=2)
        original_action = deepcopy(self.action)
        for hour in (6, 7):
            state = deepcopy(self.state)
            state["timestamp"] = f"2026-01-01T{hour:02d}:00:00"
            selected_state, selected_action = selector.select(state, self.action)
            features = normalizer.normalize(selected_state)
            features["time"] = encoder.encode(features.pop("timestamp"))
            pair = history.append(features, normalizer.normalize(selected_action))
        states, target = pair
        self.assertEqual(states[0]["time"], encoder.encode("2026-01-01T06:00:00"))
        self.assertEqual(states[1]["time"], encoder.encode("2026-01-01T07:00:00"))
        self.assertEqual(states[0]["device"]["soc_before_frac"], 0.7)
        self.assertEqual(target, {"p_net_pu": 0.01, "q_injection_pu": -0.005})
        self.assertEqual(self.action, original_action)
        self.assertEqual(json.loads(json.dumps(selector.as_dict()))["device_id"], "b2")

    def test_invalid_modes_and_unknown_bess_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Observation mode"):
            PairSelector(self.case, "invalid", "b1")
        for device_id in ("missing", "pv1"):
            with self.subTest(device=device_id), self.assertRaisesRegex(ValueError, "Unknown local BESS"):
                PairSelector(self.case, "local", device_id)
        self.case.bess = []
        with self.assertRaisesRegex(ValueError, "Unknown local BESS"):
            PairSelector(self.case, "local", "b1")
        state, action = PairSelector(self.case).select(self.state, self.action)
        self.assertEqual(state, self.state)
        self.assertEqual(action, self.action)


if __name__ == "__main__":
    unittest.main()
