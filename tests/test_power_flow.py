from pathlib import Path
import unittest

import opendssdirect

from opf_teacher.opf.single_phase import load_case
from opf_teacher.opf.single_phase.power_flow import bus_voltages


SOURCE = Path(__file__).resolve().parent / "fixtures" / "single_phase_case"


class PowerFlowTest(unittest.TestCase):
    def test_fixed_power_matches_opendss_without_changing_battery_state(self):
        case = load_case(SOURCE)
        bus = case.bus_name_to_id["load"]
        dss = opendssdirect.NewContext()
        dss.Basic.AllowChangeDir(False)
        dss.Text.Command(f'Compile "{SOURCE / "dss" / "Master.dss"}"')
        dss.Text.Command("Vsource.source.model=Ideal")
        dss.Text.Command("New Load.snapshot bus1=load.1 phases=1 kv=0.23 kw=0 kvar=0 model=1 vminpu=0 vmaxpu=2")
        for demand in (complex(2.0, 0.5), complex(-1.0, 0.2)):
            dss.Text.Command(f"Edit Load.snapshot kw={demand.real} kvar={demand.imag}")
            dss.Solution.Solve()
            dss.Circuit.SetActiveBus("load")
            measured = dss.Bus.VMagAngle()[0] / 230.0
            powers = dict.fromkeys(case.buses, 0j) | {bus: demand}
            predicted = bus_voltages(case, powers)
            if abs(predicted[bus] - measured) > 1e-6:
                self.fail("Fixed-power voltage differs from OpenDSS")
        if case.bess[0].soc_init_frac != 0.5 or case.bess[0].result is not None:
            self.fail("A voltage snapshot must not advance battery state")

    def test_no_load_preserves_transformer_tap(self):
        case = load_case(SOURCE)
        case.branches[0].tap_ratio = 1.03
        voltage = bus_voltages(case, dict.fromkeys(case.buses, 0j))
        if abs(voltage[case.bus_name_to_id["load"]] - 1.03) > 1e-10:
            self.fail("No-load voltage must include the branch tap ratio")
