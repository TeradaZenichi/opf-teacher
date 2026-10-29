from copy import deepcopy
import json
from pathlib import Path
import unittest

from normalization import PerUnitNormalizer
from opf.common import Base
from opf.single_phase import load_case
from teacher import StateActionHistory


ROOT = Path(__file__).resolve().parents[1]


class PerUnitNormalizerTest(unittest.TestCase):
    def setUp(self):
        self.normalizer = PerUnitNormalizer(Base(1000.0, 11.0))
        self.state = {
            "timestamp": "2026-01-01T00:00:00",
            "dt_h": 1.0,
            "buses": {4: {
                "v_before_pu": 1.04,
                "p_load_kw": 1500.0,
                "q_load_kvar": -200.0,
            }},
            "grid": {"buy_price_per_kwh": 0.3, "sell_price_per_kwh": 0.15},
            "bess": {"b1": {
                "soc_before_frac": 0.5,
                "previous_p_kw": -20.0,
                "previous_q_kvar": 10.0,
            }},
            "pv": {"pv1": {
                "available_kw": 260.0,
                "previous_p_kw": -0.5,
                "previous_q_kvar": -30.0,
            }},
        }

    def test_central_observation_uses_one_base_for_loads_bess_and_pv(self):
        normalized = self.normalizer.normalize(self.state)
        self.assertEqual(normalized["buses"][4], {
            "v_before_pu": 1.04, "p_load_pu": 1.5, "q_load_pu": -0.2,
        })
        self.assertEqual(normalized["bess"]["b1"], {
            "soc_before_frac": 0.5, "previous_p_pu": -0.02, "previous_q_pu": 0.01,
        })
        self.assertEqual(normalized["pv"]["pv1"], {
            "available_pu": 0.26, "previous_p_pu": -0.0005, "previous_q_pu": -0.03,
        })
        self.assertEqual(normalized["grid"], self.state["grid"])
        self.assertEqual(normalized["timestamp"], self.state["timestamp"])
        self.assertEqual(normalized["dt_h"], 1.0)
        self.assertEqual(self.normalizer.denormalize(normalized), self.state)

    def test_scalar_and_phase_actions_round_trip_to_executable_units(self):
        physical = {
            "bess": {"b1": {"p_net_kw": -20.0, "q_injection_kvar": 10.0}},
            "pv": {"pv1": {
                "generation_kw": {"a": 100.0, "b": 200.0, "c": 300.0},
                "q_injection_kvar": {"a": -10.0, "b": 0.0, "c": 20.0},
            }},
        }
        normalized = self.normalizer.normalize(physical)
        self.assertEqual(normalized["bess"]["b1"], {"p_net_pu": -0.02, "q_injection_pu": 0.01})
        self.assertEqual(normalized["pv"]["pv1"]["generation_pu"], {"a": 0.1, "b": 0.2, "c": 0.3})
        self.assertEqual(self.normalizer.denormalize(normalized), physical)

    def test_local_observation_and_phase_powers_keep_the_network_total_base(self):
        local = {
            "bus": {
                "v_before_pu": {"a": 1.04, "b": 0.99, "c": 0.98},
                "angle_before_deg": {"a": -0.1, "b": -120.1, "c": 119.9},
                "p_load_kw": {"a": 100.0, "b": 200.0, "c": 300.0},
                "q_load_kvar": {"a": 20.0, "b": 40.0, "c": 60.0},
            },
            "device": {
                "soc_before_frac": 0.5,
                "previous_p_kw": {"a": -10.0, "b": 0.0, "c": 20.0},
                "previous_q_kvar": {"a": 0.0, "b": 10.0, "c": -20.0},
            },
            "phase_order": ["a", "b", "c"],
        }
        normalized = self.normalizer.normalize(local)
        self.assertEqual(normalized["bus"]["v_before_pu"], local["bus"]["v_before_pu"])
        self.assertEqual(normalized["bus"]["angle_before_deg"], local["bus"]["angle_before_deg"])
        self.assertEqual(normalized["bus"]["p_load_pu"], {"a": 0.1, "b": 0.2, "c": 0.3})
        self.assertAlmostEqual(sum(normalized["bus"]["p_load_pu"].values()), 600.0 / 1000.0)
        self.assertEqual(normalized["device"]["previous_p_pu"], {"a": -0.01, "b": 0.0, "c": 0.02})
        self.assertEqual(self.normalizer.denormalize(normalized), local)

    def test_normalization_copies_data_and_does_not_divide_pu_twice(self):
        original = deepcopy(self.state)
        normalized = self.normalizer.normalize(self.state)
        self.assertEqual(self.state, original)
        self.assertEqual(self.normalizer.normalize(normalized), normalized)
        physical = self.normalizer.denormalize(normalized)
        physical["grid"]["buy_price_per_kwh"] = 0.9
        normalized["buses"][4]["v_before_pu"] = 0.8
        self.assertEqual(self.state, original)
        self.assertEqual(normalized["grid"]["buy_price_per_kwh"], 0.3)

    def test_case_configuration_supplies_the_bases_and_metadata(self):
        config = json.loads((ROOT / "examples" / "case5_local" / "config.json").read_text(encoding="utf-8"))
        case = load_case(ROOT / "examples" / "case5_local")
        normalizer = PerUnitNormalizer(case.base)
        self.assertEqual(normalizer.as_dict()["base"], config["base"])
        metadata = json.loads(json.dumps(normalizer.as_dict()))
        restored = PerUnitNormalizer(Base(**metadata["base"]))
        self.assertEqual(restored.normalize(self.state), normalizer.normalize(self.state))
        case.base.s_base_kva = 2000.0
        self.assertEqual(normalizer.normalize(self.state)["bess"]["b1"]["previous_p_pu"], -0.02)
        changed = PerUnitNormalizer(case.base)
        self.assertEqual(changed.normalize(self.state)["bess"]["b1"]["previous_p_pu"], -0.01)

    def test_temporal_positions_share_the_same_scale_and_current_target(self):
        history = StateActionHistory(window=3)
        action = {"bess": {"b1": {"p_net_kw": -20.0, "q_injection_kvar": 10.0}}}
        for time in range(3):
            state = deepcopy(self.state)
            state["timestamp"] = f"2026-01-01T0{time}:00:00"
            pair = history.append(self.normalizer.normalize(state), self.normalizer.normalize(action))
            if time < 2:
                self.assertIsNone(pair)
        states, current_action = pair
        for state in states:
            self.assertEqual(state["bess"]["b1"]["previous_p_pu"], -0.02)
            self.assertEqual(state["buses"][4]["p_load_pu"], 1.5)
        self.assertEqual(current_action["bess"]["b1"]["p_net_pu"], -0.02)

    def test_bases_must_be_finite_and_positive(self):
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(base="power", value=invalid), self.assertRaisesRegex(ValueError, "s_base_kva"):
                Base(invalid, 11.0)
            with self.subTest(base="voltage", value=invalid), self.assertRaisesRegex(ValueError, "v_base_kv"):
                Base(1000.0, invalid)


if __name__ == "__main__":
    unittest.main()
