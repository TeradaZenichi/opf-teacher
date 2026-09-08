from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyomo.environ as pyo

from opf import BessState, BusState, PvState, load_case
from teacher import BessOpt, Teacher


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
    def setUp(self):
        self.teacher = Teacher("examples/case5", active_power_only=True)
        self.case = self.teacher.case

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
        local_x, local_y = bess.state_action(solved.buses[bess.bus])
        pv_x, pv_y = pv.state_action(solved.buses[pv.bus])

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
            extra.state_action(self.case.buses[4])

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
        with patch("teacher.pyo.SolverFactory") as factory:
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
