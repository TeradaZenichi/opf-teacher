from __future__ import annotations

import unittest

from opf.active_model import solve_active_power_opf
from opf.data import load_case
from opf.export import branches_dataframe, buses_dataframe, devices_dataframe


class ActivePowerOpfTest(unittest.TestCase):
    def test_case5_solves_and_exports_active_device_dispatch(self):
        case = solve_active_power_opf(load_case("examples/case5"))
        devices = devices_dataframe(case)
        buses = buses_dataframe(case)
        branches = branches_dataframe(case)

        self.assertEqual(case.summary.status, "optimal")
        self.assertEqual(len(devices), 24)
        self.assertEqual(len(buses), 24)
        self.assertEqual(len(branches), 24)
        self.assertIn("bess_b1_p_net_kw", devices)
        self.assertIn("pv_pv1_p_net_kw", devices)
        self.assertAlmostEqual(
            float(devices["bess_b1_p_net_kw"].iloc[0]),
            (
                float(case.bess[0].result.charge_kw.iloc[0])
                - float(case.bess[0].result.discharge_kw.iloc[0])
            ),
        )
        self.assertAlmostEqual(
            float(devices["pv_pv1_p_net_kw"].iloc[-1]),
            -float(case.pv[0].result.p_net_kw.iloc[-1]),
        )
        self.assertIn("bus_004_voltage_pu", buses)
        self.assertIn("bus_004_load_current_a", buses)
        self.assertIn("line_001_002_p_kw", branches)
        self.assertIn("line_001_002_current_a", branches)
        self.assertAlmostEqual(float(case.bess[0].result.q_kvar.abs().max()), 0.0)
        self.assertAlmostEqual(float(case.pv[0].result.q_kvar.abs().max()), 0.0)
        self.assertAlmostEqual(
            float(case.bess[0].result.soc_kwh.iloc[-1]),
            case.bess[0].soc_init_frac * case.bess[0].e_cap_kwh,
        )


if __name__ == "__main__":
    unittest.main()
