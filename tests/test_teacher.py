from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyomo.environ as pyo

from opf_teacher.opf.single_phase import BessState, BusState, PvState, load_case
from opf_teacher.opf.single_phase.states import local_state_action
from opf_teacher.teacher import BessOpt, SolveConfig, Teacher


def observations(case, soc=0.7):
    return {
        "buses": {
            bid: BusState(1.01, float(bus.p_load_kw.iloc[0]) + 1.0,
                          float(bus.q_load_kw.iloc[0]) + 0.5)
            for bid, bus in case.buses.items()
        },
        "bess": {d.id: BessState(soc, -2.0, 3.0) for d in case.bess},
        "pv": {d.id: PvState(10.0, 0.0, 0.0) for d in case.pv},
    }


class TeacherTest(unittest.TestCase):
    def test_public_pairs_are_pu_and_owned_model_results_keep_explicit_units(self):
        self.teacher.observe(**observations(self.case))
        solved = self.teacher.solve()
        state, action = self.teacher.state_action()
        self.teacher.normalizer.require_pu(state)
        self.teacher.normalizer.require_pu(action)
        self.assertNotIn("p_net_kw", action["bess"]["b1"])
        expected = solved.bess[0].action.p_net_kw
        self.assertAlmostEqual(action["bess"]["b1"]["p_net_pu"], expected / solved.base.s_base_kva)
        physical = self.teacher.normalizer.denormalize(action)
        self.assertAlmostEqual(physical["bess"]["b1"]["p_net_kw"], expected)
        self.assertEqual(state["bess"]["b1"]["soc_before_frac"], 0.7)

    def test_named_observations_require_pu_and_matching_bases(self):
        valid = {
            "observation_schema_version": 2, "units": "pu",
            "phase_power_base": "total_network",
            "base": self.teacher.normalizer.as_dict()["base"],
        }
        with self.assertRaisesRegex(ValueError, "units"):
            self.teacher.observe_dict({**valid, "units": "physical"})
        with self.assertRaisesRegex(ValueError, "s_base_kva"):
            self.teacher.observe_dict({**valid, "base": {"s_base_kva": 2000.0, "v_base_kv": 11.0}})
        with self.assertRaisesRegex(ValueError, "physical field"):
            self.teacher.observe_dict({**valid, "bess": {"b1": {"previous_p_kw": -20.0}}})

    def test_horizon_preserves_source_and_aligns_all_profiles(self):
        original = load_case("examples/case5")
        original.summary = object()
        original.bess[0].action = object()
        horizon = original.horizon(start=1, periods=3)
        self.assertTrue(horizon.index.equals(original.index[1:4]))
        self.assertTrue(horizon.price.index.equals(horizon.index))
        for bus in horizon.buses.values():
            self.assertTrue(bus.p_load_kw.index.equals(horizon.index))
            self.assertTrue(bus.q_load_kw.index.equals(horizon.index))
        self.assertTrue(horizon.pv[0].avail_kw.index.equals(horizon.index))
        self.assertIsNone(horizon.summary)
        self.assertIsNone(horizon.bess[0].action)
        self.assertIsNotNone(original.summary)
        original_soc = original.bess[0].soc_init_frac
        horizon.bess[0].soc_init_frac = 0.7
        self.assertEqual(original.bess[0].soc_init_frac, original_soc)
        self.assertEqual(original.n_periods, 24)

    def test_horizon_rejects_invalid_ranges(self):
        case = load_case("examples/case5")
        for start in (-1, 24, True, 1.5):
            with self.subTest(start=start), self.assertRaises(ValueError):
                case.horizon(start)
        for periods in (0, -1, True, 1.5, 25):
            with self.subTest(periods=periods), self.assertRaises(ValueError):
                case.horizon(0, periods)

    def setUp(self):
        self.teacher = Teacher("examples/case5", formulation="single_phase_active")
        self.case = self.teacher.case

    def test_teachers_do_not_modify_or_share_the_input_case(self):
        original = load_case("examples/case5")
        first = Teacher(original, formulation="single_phase_active")
        second = Teacher(original, formulation="single_phase_active")
        original_load = original.buses[4].p_load_kw.copy()
        original_soc = original.bess[0].soc_init_frac

        first.observe(**observations(first.case)).build()
        solved = first.solve(config=SolveConfig())

        self.assertIs(solved, first.case)
        self.assertIsNot(first.case, original)
        self.assertIsNot(first.case, second.case)
        self.assertTrue(original.buses[4].p_load_kw.equals(original_load))
        self.assertEqual(original.bess[0].soc_init_frac, original_soc)
        self.assertEqual(second.case.bess[0].soc_init_frac, original_soc)
        self.assertIsNone(original.summary)
        self.assertIsNone(second.case.bess[0].action)

    def test_config_and_legacy_options_cannot_be_mixed(self):
        with self.assertRaisesRegex(ValueError, "config or legacy"):
            self.teacher.solve(config=SolveConfig(), solver="appsi_highs")

    def test_observation_clears_previous_quality(self):
        self.case.quality = {"solution_quality": "old"}
        self.teacher.observe(**observations(self.case))
        self.assertIsNone(self.case.quality)

    def test_local_and_central_pairs_use_pre_action_observations(self):
        observed = observations(self.case)
        self.teacher.observe(**observed)
        # External changes must not alter the captured observation.
        observed["bess"]["b1"].soc_before_frac = 0.2
        solved = self.teacher.solve()
        self.assertIs(solved, self.case)
        x, y = solved.state_action()
        bess = solved.bess[0]
        pv = solved.pv[0]
        local_x, local_y = local_state_action(bess, solved.buses[bess.bus])
        pv_x, pv_y = local_state_action(pv, solved.buses[pv.bus])

        self.assertEqual(x["timestamp"], solved.index[0].isoformat())
        self.assertEqual(x["bess"]["b1"]["soc_before_frac"], 0.7)
        self.assertEqual(local_x["bus"]["v_before_pu"], 1.01)
        self.assertNotAlmostEqual(solved.buses[bess.bus].result.v_pu.iloc[0], 1.01)
        self.assertNotAlmostEqual(bess.result.soc_frac.iloc[0], 0.7)
        self.assertEqual(set(local_x), {"bus", "device"})
        self.assertEqual(y["bess"]["b1"], local_y)
        self.assertEqual(y["pv"]["pv1"], pv_y)
        self.assertEqual(pv_x["device"]["available_kw"], 10.0)
        self.assertAlmostEqual(local_y["p_net_kw"], bess.result.p_net_kw.iloc[0])
        self.assertAlmostEqual(pv_y["generation_kw"], pv.result.gen_kw.iloc[0])
        self.assertGreater(pv_y["generation_kw"], 0.0)
        self.assertEqual(local_y["q_injection_kvar"], 0.0)
        self.assertNotIn("grid", y)
        # Preserves the original terminal target despite observing a new SoC.
        self.assertAlmostEqual(bess.result.soc_frac.iloc[-1], 0.5)
        gain = (bess.result.charge_kw.iloc[0] * bess.eta_charge
                - bess.result.discharge_kw.iloc[0] / bess.eta_discharge)
        self.assertAlmostEqual(bess.result.soc_frac.iloc[0],
                               0.7 + gain * solved.dt_h / bess.e_cap_kwh)
        x["bess"]["b1"]["soc_before_frac"] = 0.1
        self.assertEqual(solved.state_action()[0]["bess"]["b1"]["soc_before_frac"], 0.7)

    def test_named_environment_observation_is_accepted_directly(self):
        case = self.case
        observation = {
            "observation_schema_version": 1,
            "timestamp": case.index[0].isoformat(),
            "dt_h": case.dt_h,
            "phase_order": ["a"],
            "buses": {
                bus.name: {
                    "v_before_pu": {"a": 1.0},
                    "angle_before_deg": {"a": 0.0},
                    "p_load_kw": {"a": float(bus.p_load_kw.iloc[0])},
                    "q_load_kvar": {"a": float(bus.q_load_kw.iloc[0])},
                }
                for bus in case.buses.values()
            },
            "grid": {
                "buy_price_per_kwh": float(case.price.iloc[0]),
                "sell_price_per_kwh": float(case.price.iloc[0]) * case.grid.feed_in_ratio,
            },
            "bess": {
                device.id: {
                    "soc_before_frac": device.soc_init_frac,
                    "previous_p_kw": {"a": 0.0},
                    "previous_q_kvar": {"a": 0.0},
                }
                for device in case.bess
            },
            "pv": {
                device.id: {
                    "available_kw": {"a": float(device.avail_kw.iloc[0])},
                    "previous_p_kw": {"a": 0.0},
                    "previous_q_kvar": {"a": 0.0},
                }
                for device in case.pv
            },
        }

        observation = self.teacher.normalizer.normalize(observation)
        observation.update({
            "observation_schema_version": 2, "units": "pu",
            "base": self.teacher.normalizer.as_dict()["base"],
            "phase_power_base": "total_network",
        })

        solved = self.teacher.observe_dict(observation).solve()
        x, _ = solved.state_action()

        self.assertEqual(x["timestamp"], observation["timestamp"])
        self.assertEqual(x["bess"]["b1"]["soc_before_frac"], 0.5)

    def test_named_observation_version_is_required(self):
        with self.assertRaisesRegex(ValueError, "observation_schema_version"):
            self.teacher.observe_dict({"observation_schema_version": 1})

    def test_multiple_devices_and_local_bus_validation(self):
        extra = deepcopy(self.case.bess[0])
        extra.id, extra.bus = "b2", 3
        self.case.bess.insert(0, extra)
        extra_pv = deepcopy(self.case.pv[0])
        extra_pv.id, extra_pv.bus = "pv2", 2
        self.case.pv.insert(0, extra_pv)
        self.teacher.observe(**observations(self.case)).solve()
        x, y = self.case.state_action()
        self.assertEqual(list(y["bess"]), ["b1", "b2"])
        self.assertEqual(list(y["pv"]), ["pv1", "pv2"])
        self.assertEqual(len(x["buses"]), 5)
        with self.assertRaisesRegex(ValueError, "not connected"):
            local_state_action(extra, self.case.buses[4])

    def test_observation_invalidates_previous_model_and_labels(self):
        self.teacher.observe(**observations(self.case)).solve()
        old_model = self.teacher.model
        self.teacher.observe(**observations(self.case, soc=0.6))
        self.assertIsNone(self.teacher.model)
        self.assertIsNone(self.case.bess[0].action)
        self.assertIsNone(self.case.buses[4].result)
        with self.assertRaisesRegex(ValueError, "solve"):
            self.case.state_action()
        self.teacher.solve()
        self.assertIsNot(self.teacher.model, old_model)
        self.assertEqual(self.case.state_action()[0]["bess"]["b1"]["soc_before_frac"], 0.6)

    def test_empty_device_groups_and_fractional_observations(self):
        self.case.bess = []
        self.case.pv = []
        self.case.buses[1].p_load_kw = self.case.buses[1].p_load_kw.astype(int)
        observed = observations(self.case)
        observed["buses"][1].p_load_kw = 0.25
        self.teacher.observe(**observed).solve()
        x, y = self.case.state_action()
        self.assertEqual(x["buses"][1]["p_load_kw"], 0.25)
        self.assertEqual(y, {"bess": {}, "pv": {}})
        self.assertEqual(self.case.buses[1].p_load_kw.iloc[0], 0.25)

    def test_rejects_missing_nonfinite_and_out_of_range_observations(self):
        for kind in ("missing", "nan", "soc"):
            observed = observations(self.case)
            if kind == "missing":
                observed["buses"].pop(1)
            elif kind == "nan":
                observed["buses"][1].v_before_pu = float("nan")
            else:
                observed["bess"]["b1"].soc_before_frac = 1.1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.teacher.observe(**observed)
            self.assertIsNone(self.case.buses[1].state)
            self.assertEqual(self.case.bess[0].soc_init_frac, 0.5)

    def test_no_implicit_observation_from_optimized_results(self):
        self.teacher.solve()
        with self.assertRaisesRegex(ValueError, "observation"):
            self.case.state_action()

    def test_failed_resolve_does_not_leave_previous_actions(self):
        self.teacher.observe(**observations(self.case)).solve()
        failed = SimpleNamespace(solver=SimpleNamespace(termination_condition="infeasible"))
        with patch("opf_teacher.teacher.pyo.SolverFactory") as factory:
            factory.return_value.solve.return_value = failed
            with self.assertRaisesRegex(RuntimeError, "infeasible"):
                self.teacher.solve()
        self.assertIsNone(self.case.bess[0].action)
        self.assertIsNone(self.case.summary)

    def test_legacy_api_builds_full_model_and_explicit_terminal_target(self):
        legacy = BessOpt("examples/case5")
        self.assertIsInstance(legacy, Teacher)
        legacy.case.bess[0].soc_terminal_frac = 0.8
        legacy.case.bess[0].cyclic_soc = False
        legacy.build()
        self.assertTrue(hasattr(legacy.model, "socp"))
        self.assertAlmostEqual(pyo.value(legacy.model.soc_terminal["b1"].lower), 0.08)
        self.case.bess[0].soc_terminal_frac = 0.8
        self.case.bess[0].cyclic_soc = False
        self.teacher.observe(**observations(self.case)).solve()
        self.assertAlmostEqual(self.case.bess[0].result.soc_frac.iloc[-1], 0.8)


if __name__ == "__main__":
    unittest.main()
