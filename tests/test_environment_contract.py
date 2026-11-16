import unittest
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import pyomo.environ as pyo

from opf_teacher.normalization import PUNormalizer, TimeEncoder
from opf_teacher.teacher import PairSelector, Teacher

try:
    from opendss_environment import EnvironmentConfig, MicrogridEnv
except ImportError:
    MicrogridEnv = None


@unittest.skipIf(MicrogridEnv is None, "Install the environment extra to test the integration")
class EnvironmentContractTest(unittest.TestCase):
    def test_environment_versions_preserve_bess_observations(self):
        source = "tests/fixtures/single_phase_case"
        env = MicrogridEnv(EnvironmentConfig(source, 2))
        self.addCleanup(env.close)
        observation, _ = env.reset()
        for version in (2, 4):
            teacher = Teacher(source, formulation="single_phase_active")
            teacher.observe_dict(observation | {"observation_schema_version": version})
            if teacher.case.bess[0].soc_init_frac != observation["bess"]["b1"]["soc_before_frac"]:
                raise ValueError("Environment versions must preserve the observed SoC")
        with self.assertRaisesRegex(NotImplementedError, "EVSE"):
            teacher.observe_dict(observation | {"evse": {"c1": {}}})

    def test_evse_cases_are_not_silently_optimized_without_their_chargers(self):
        for name, formulation in (("single_phase_case", "single_phase_socp"),
                                  ("three_phase_transformer_case", "three_phase_ivr")):
            with self.subTest(formulation=formulation), TemporaryDirectory() as directory:
                case = Path(directory) / "case"
                shutil.copytree(Path("tests/fixtures") / name, case)
                devices_path = case / "devices.json"
                devices = json.loads(devices_path.read_text(encoding="utf-8"))
                devices["evse"] = [{"id": "c1"}]
                devices_path.write_text(json.dumps(devices), encoding="utf-8")
                with self.assertRaisesRegex(NotImplementedError, "EVSE"):
                    Teacher(case, formulation=formulation)

    def test_real_single_phase_voltage_and_current_bases(self):
        teacher = Teacher("tests/fixtures/single_phase_case", formulation="single_phase_active")
        case = teacher.case
        self.assertAlmostEqual(case.buses[case.root].kv_base_ln, 0.23)
        self.assertAlmostEqual(case.branches[0].s_max_kva, 23.0)
        env = MicrogridEnv(EnvironmentConfig("tests/fixtures/single_phase_case", 2))
        observation, _ = env.reset()
        self.assertEqual(observation["network"]["representation"], "single_phase")
        self.assertAlmostEqual(observation["buses"]["source"]["v_before_pu"]["a"], 1.0, places=5)
        teacher.observe_dict(observation)

    def test_both_libraries_can_run_from_a_third_repository_data_folder(self):
        cases = (
            ("tests/fixtures/single_phase_case", "single_phase_active"),
            ("examples/case5_local", "single_phase_active"),
            ("examples/case5_unbalanced", "three_phase_ivr"),
        )
        with TemporaryDirectory() as directory:
            for original_source, formulation in cases:
                with self.subTest(source=original_source):
                    source = Path(directory) / "data" / Path(original_source).name
                    shutil.copytree(original_source, source)
                    original = Teacher(source, formulation=formulation).case.horizon(0, periods=2)
                    env = MicrogridEnv(EnvironmentConfig(source, 2))
                    observation, _ = env.reset()
                    for step in original.periods:
                        teacher = Teacher(original.horizon(step), formulation=formulation)
                        teacher.observe_dict(observation)
                        solved = teacher.solve()
                        _, action = teacher.state_action()
                        before = observation
                        observation, _, _, _, info = env.step(action)
                        self.assertEqual(info["observation_before"], before)
                        self.assertEqual(info["requested_action"], action)
                        self.assertEqual(len(env.bess_list[0].array_soc), step + 1)
                        self.assertAlmostEqual(env.bess_list[0].soc, float(solved.bess[0].result.soc_frac.iloc[0]), places=7)
                    self.assertIsNone(observation)

    def test_socp_actions_run_in_the_environment(self):
        if not pyo.SolverFactory("gurobi_direct").available(exception_flag=False):
            self.skipTest("SOCP integration requires a compatible solver")
        source = "tests/fixtures/single_phase_case"
        original = Teacher(source, formulation="single_phase_socp").case.horizon(0, periods=2)
        original.bess[0].cyclic_soc = True
        env = MicrogridEnv(EnvironmentConfig(source, 2))
        observation, _ = env.reset()
        for step in original.periods:
            teacher = Teacher(original.horizon(step), formulation="single_phase_socp")
            teacher.observe_dict(observation)
            teacher.solve()
            _, action = teacher.state_action()
            observation, _, _, _, info = env.step(action)
            self.assertEqual(info["requested_action"], action)

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
