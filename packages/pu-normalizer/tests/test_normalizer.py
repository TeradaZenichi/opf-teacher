from copy import deepcopy
import unittest

from pu_normalizer import PUNormalizer


class PUNormalizerTest(unittest.TestCase):
    def test_aggregate_and_phase_powers_share_the_network_base(self):
        normalizer = PUNormalizer(1000.0, 11.0)
        physical = {
            "v_before_pu": 1.08,
            "p_load_kw": {"a": 100.0, "b": 200.0, "c": 300.0},
            "p_net_kw": -20.0,
            "grid_import_kw": 1500.0,
            "measurements": {"generation_total_kw": -0.5, "q_injection_total_kvar": 10.0},
        }
        original = deepcopy(physical)
        normalized = normalizer.normalize(physical)
        self.assertEqual(normalized["p_load_pu"], {"a": 0.1, "b": 0.2, "c": 0.3})
        self.assertEqual(normalized["p_net_pu"], -0.02)
        self.assertEqual(normalized["grid_import_pu"], 1.5)
        self.assertEqual(normalized["v_before_pu"], 1.08)
        self.assertEqual(normalized["measurements"]["generation_total_pu"], -0.0005)
        self.assertEqual(normalizer.denormalize(normalized), physical)
        self.assertEqual(normalizer.normalize(normalized), normalized)
        self.assertEqual(physical, original)

    def test_pu_interface_rejects_physical_fields_and_wrong_bases(self):
        normalizer = PUNormalizer(1000.0, 11.0)
        normalizer.require_pu({"bess": {"b1": {"p_net_pu": -0.02}}})
        with self.assertRaisesRegex(ValueError, "physical field"):
            normalizer.require_pu({"bess": {"b1": {"p_net_kw": -20.0}}})
        normalizer.check_base({"s_base_kva": 1000.0, "v_base_kv": 11.0})
        with self.assertRaisesRegex(ValueError, "s_base_kva"):
            normalizer.check_base({"s_base_kva": 2000.0, "v_base_kv": 11.0})

    def test_nonfinite_powers_and_invalid_bases_are_rejected(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(base=value), self.assertRaises(ValueError):
                PUNormalizer(value, 11.0)
        normalizer = PUNormalizer(1000.0, 11.0)
        for value in (float("nan"), float("inf")):
            with self.subTest(power=value), self.assertRaisesRegex(ValueError, "finite"):
                normalizer.denormalize({"p_net_pu": {"a": value}})


if __name__ == "__main__":
    unittest.main()
