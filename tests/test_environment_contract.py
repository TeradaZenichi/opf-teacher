import unittest

from normalization import PUNormalizer, TimeEncoder
from teacher import PairSelector, Teacher

try:
    from opendss_env import EnvironmentConfig, MicrogridEnv
except ImportError:
    MicrogridEnv = None


@unittest.skipIf(MicrogridEnv is None, "Install the environment extra to test the integration")
class EnvironmentContractTest(unittest.TestCase):
    def test_returned_observations_feed_the_teacher_in_both_formulations(self):
        cases = (
            ("examples/case5_local", "single_phase_active"),
            ("examples/case5_unbalanced", "three_phase_ivr"),
        )
        for source, formulation in cases:
            with self.subTest(formulation=formulation):
                original = Teacher(source, formulation=formulation).case.horizon(0, periods=2)
                env = MicrogridEnv(EnvironmentConfig(source, original.n_periods))
                observation, _ = env.reset()
                normalizer = PUNormalizer(original.base.s_base_kva, original.base.v_base_kv)
                self.assertIs(type(normalizer), type(env.normalizer))
                self.assertEqual(observation["units"], "pu")
                encoder = TimeEncoder()
                for step in original.periods:
                    before = observation
                    teacher = Teacher(original.horizon(step), formulation=formulation)
                    teacher.observe_dict(before)
                    solved = teacher.solve()
                    state, action = teacher.state_action()
                    self.assertEqual(state["timestamp"], before["timestamp"])
                    for mode in ("central", "local"):
                        selected_state, selected_action = PairSelector(original, mode, "b1").select(state, action)
                        features = selected_state
                        features["time"] = encoder.encode(features.pop("timestamp"))
                        self.assertEqual(features["time"], encoder.encode(before["timestamp"]))
                        soc = features["bess"]["b1"]["soc_before_frac"] if mode == "central" else features["device"]["soc_before_frac"]
                        self.assertEqual(soc, before["bess"]["b1"]["soc_before_frac"])
                        target = selected_action
                        physical = normalizer.denormalize(target)
                        requested = physical["bess"]["b1"] if mode == "central" else physical
                        expected = solved.bess[0].action.p_net_kw
                        if isinstance(expected, dict):
                            for phase, value in expected.items():
                                self.assertAlmostEqual(requested["p_net_kw"][phase], value)
                        else:
                            self.assertAlmostEqual(requested["p_net_kw"], expected)
                    observation, _, terminated, truncated, info = env.step(action)
                    self.assertEqual(info["observation_before"], before)
                    self.assertEqual(len(env.bess_list[0].array_soc), step + 1)
                    self.assertFalse(truncated)
                    if terminated:
                        self.assertIsNone(observation)
                    else:
                        self.assertEqual(observation, env.observe())
                        self.assertEqual(observation["timestamp"], original.index[step + 1].isoformat())
                env.close()


if __name__ == "__main__":
    unittest.main()
