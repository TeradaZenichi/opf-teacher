import unittest

import numpy as np

from opf_teacher.opf.single_phase.data import load_case as load_single_phase_case
from opf_teacher.opf.three_phase.data import load_case
from opf_teacher.opf.three_phase.ivr_model import DispatchEvaluation, IVRModel
from opf_teacher.teacher import ThreePhaseTeacher


class ThreePhaseDataTest(unittest.TestCase):
    def test_dispatch_evaluation_exposes_physical_results_as_fields(self):
        case = load_case("tests/fixtures/three_phase_transformer_case")
        model = IVRModel(case)
        bounds, initial = model._decision_bounds_and_start()
        evaluated = model._evaluate_dispatch(initial)
        self.assertIsInstance(evaluated, DispatchEvaluation)
        self.assertEqual(len(evaluated.voltages), case.n_periods)
        self.assertLess(evaluated.residual, 1e-8)
        self.assertGreater(evaluated.objective, 0.0)
        self.assertIs(model._evaluate_dispatch(initial.copy()), evaluated)

    def test_horizon_keeps_phase_profiles_aligned_and_the_source_unchanged(self):
        original = load_case("examples/case5_unbalanced")
        horizon = original.horizon(start=1, periods=2)
        self.assertEqual(horizon.n_periods, 2)
        self.assertTrue(horizon.index.equals(original.index[1:3]))
        self.assertTrue(horizon.price.index.equals(horizon.index))
        for bus in horizon.buses.values():
            for phase in bus.phases:
                self.assertTrue(bus.p_load_kw[phase].index.equals(horizon.index))
                self.assertTrue(bus.q_load_kvar[phase].index.equals(horizon.index))
        for connected in horizon.pv:
            self.assertTrue(connected.device.avail_kw.index.equals(horizon.index))
            for values in connected.available_kw.values():
                self.assertTrue(values.index.equals(horizon.index))
        original_load = original.buses[2].p_load_kw["a"].copy()
        horizon.buses[2].p_load_kw["a"].iloc[0] = 999.0
        self.assertTrue(original.buses[2].p_load_kw["a"].equals(original_load))
        self.assertEqual(original.n_periods, 24)
        self.assertEqual(original.horizon(start=23).n_periods, 1)

    def test_horizon_clears_observations_actions_and_results(self):
        original = load_case("examples/case5_unbalanced")
        original.buses[2].state = object()
        original.buses[2].result = object()
        original.bess[0].action = object()
        original.pv[0].result = object()
        original.grid.state = object()
        original.summary = object()
        original.quality = {"solution_quality": "old"}
        horizon = original.horizon(start=1)
        self.assertIsNone(horizon.buses[2].state)
        self.assertIsNone(horizon.buses[2].result)
        self.assertIsNone(horizon.bess[0].action)
        self.assertIsNone(horizon.pv[0].result)
        self.assertIsNone(horizon.grid.state)
        self.assertIsNone(horizon.summary)
        self.assertIsNone(horizon.quality)
        self.assertIsNotNone(original.summary)

    def test_horizon_rejects_invalid_ranges(self):
        case = load_case("examples/case5_unbalanced")
        for start in (-1, 24, True, 1.5):
            with self.subTest(start=start), self.assertRaises(ValueError):
                case.horizon(start=start)
        for periods in (0, -1, True, 1.5, 25):
            with self.subTest(periods=periods), self.assertRaises(ValueError):
                case.horizon(start=0, periods=periods)

    def test_case5_unbalanced_preserves_bus_totals(self):
        balanced = load_single_phase_case("examples/case5")
        unbalanced = load_case("examples/case5_unbalanced")

        self.assertEqual(unbalanced.n_periods, balanced.n_periods)
        self.assertEqual(len(unbalanced.buses), 5)
        self.assertEqual(len(unbalanced.branches), 4)
        self.assertEqual(unbalanced.bus_name_to_id["bus_002"], 2)
        for bus_id in balanced.buses:
            np.testing.assert_allclose(
                unbalanced.buses[bus_id].p_load_total_kw.to_numpy(),
                balanced.buses[bus_id].p_load_kw.to_numpy(),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                unbalanced.buses[bus_id].q_load_total_kvar.to_numpy(),
                balanced.buses[bus_id].q_load_kw.to_numpy(),
                atol=1e-12,
            )

    def test_only_demand_is_unbalanced(self):
        case = load_case("examples/case5_unbalanced")

        self.assertEqual(case.grid.v_ref_pu, {"a": 1.0, "b": 1.0, "c": 1.0})
        self.assertEqual(case.grid.angle_ref_deg, {"a": 0.0, "b": -120.0, "c": 120.0})
        self.assertEqual(case.bess[0].connection.phases, ("a", "b", "c"))
        self.assertEqual(case.bess[0].connection.dispatch_mode, "aggregate")
        self.assertEqual(case.pv[0].connection.phases, ("a", "b", "c"))
        for branch in case.branches:
            impedance = np.asarray(branch.z_matrix_ohm)
            np.testing.assert_allclose(impedance, impedance.T)
            np.testing.assert_allclose(np.diag(impedance), impedance[0, 0])
            mutual = impedance[~np.eye(len(impedance), dtype=bool)]
            np.testing.assert_allclose(mutual, mutual[0])
            self.assertEqual(len(set(branch.norm_amps.values())), 1)
        for phase in ("a", "b", "c"):
            np.testing.assert_allclose(
                case.pv[0].available_kw[phase].to_numpy(),
                (case.pv[0].device.avail_kw / 3.0).to_numpy(),
                atol=1e-12,
            )
        self.assertNotEqual(
            case.buses[2].p_load_kw["a"].iloc[0],
            case.buses[2].p_load_kw["c"].iloc[0],
        )

    def test_transformer_and_multiple_voltage_levels_are_solved(self):
        case = load_case("tests/fixtures/three_phase_transformer_case")

        self.assertEqual(
            [branch.element_type for branch in case.branches],
            ["transformer", "line"],
        )
        self.assertGreater(case.buses[1].kv_base_ln, case.buses[2].kv_base_ln)
        solved = ThreePhaseTeacher(case).solve()

        self.assertLess(solved.quality["max_power_flow_residual_a"], 1e-5)
        self.assertGreater(solved.buses[2].result.v_pu["a"].iloc[0], 1.0)
        self.assertGreater(solved.buses[3].result.v_pu["c"].iloc[0], 0.9)

    def test_delta_wye_transformer_is_solved_with_phase_shift(self):
        case = load_case(
            "tests/fixtures/three_phase_transformer_case/config_delta.json"
        )

        self.assertEqual(case.branches[0].connections, ("delta", "wye"))
        solved = ThreePhaseTeacher(case).solve()

        self.assertLess(solved.quality["max_power_flow_residual_a"], 1e-5)
        low_angle = solved.buses[2].result.angle_deg["a"].iloc[0]
        self.assertGreater(abs(low_angle), 20.0)


if __name__ == "__main__":
    unittest.main()
