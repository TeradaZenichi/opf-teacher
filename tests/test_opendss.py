from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyomo.environ as pyo

from opf.data import load_case
from opf.model import build_model
from opf.opendss import load_opendss_network, resolve_bus_id
from opf.results import analyze_socp_gap


class _FakeError:
    error_code = 0
    error_desc = ""


class _FakeDSSInterface:
    num_circuits = 1


class _FakeCircuit:
    def __init__(self, owner, buses):
        self._owner = owner
        self.buses_names = list(buses)

    def set_active_bus(self, name):
        self._owner.active_bus = name
        return 0


class _FakeBus:
    def __init__(self, owner, records):
        self._owner = owner
        self._records = records

    @property
    def nodes(self):
        return self._records[self._owner.active_bus]["nodes"]

    @property
    def kv_base(self):
        return self._records[self._owner.active_bus]["kv_base"]


class _FakeLines:
    def __init__(self, owner, records):
        self._owner = owner
        self._records = {record["name"]: record for record in records}
        self.names = list(self._records)
        self._name = self.names[0]

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    def __getattr__(self, item):
        return self._records[self._name][item]


class FakeDSS:
    def __init__(self):
        self.active_bus = None
        self.commands = []
        buses = {
            "bus_001": {"nodes": [1, 2, 3], "kv_base": 0.127},
            "bus_002": {"nodes": [1, 2, 3], "kv_base": 0.127},
            "bus_003": {"nodes": [1, 2, 3], "kv_base": 0.127},
        }
        matrix = [0.63, 0.10, 0.10, 0.10, 0.63, 0.10, 0.10, 0.10, 0.63]
        lines = [
            {
                "name": "line_001_002", "bus1": "bus_001.1.2.3",
                "bus2": "bus_002.1.2.3", "phases": 3, "length": 0.025,
                "units": 3, "r1": 0.63, "x1": 0.12,
                "rmatrix": matrix, "xmatrix": [v / 5 for v in matrix],
                "norm_amps": 100.0,
            },
            {
                "name": "line_003_002", "bus1": "bus_003.1.2.3",
                "bus2": "bus_002.1.2.3", "phases": 3, "length": 0.025,
                "units": 3, "r1": 0.63, "x1": 0.12,
                "rmatrix": matrix, "xmatrix": [v / 5 for v in matrix],
                "norm_amps": 100.0,
            },
        ]
        self.errorinterface = _FakeError()
        self.dssinterface = _FakeDSSInterface()
        self.circuit = _FakeCircuit(self, buses)
        self.bus = _FakeBus(self, buses)
        self.lines = _FakeLines(self, lines)

    def text(self, command):
        self.commands.append(command)
        return ""


class OpenDSSImporterTest(unittest.TestCase):
    def test_imports_positive_sequence_and_orients_radial_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            master = Path(tmp) / "Master.dss"
            master.write_text("! fake model", encoding="utf-8")
            fake = FakeDSS()
            network = load_opendss_network(master, slack_bus="bus_001", dss=fake)

        self.assertEqual(network.root_bus, 1)
        self.assertEqual(network.bus_name_to_id["bus_003"], 3)
        self.assertEqual([(b.from_bus, b.to_bus) for b in network.branches], [(1, 2), (2, 3)])
        first = network.branches[0]
        self.assertAlmostEqual(first.r_ohm, 0.63 * 0.025)
        self.assertAlmostEqual(first.x_ohm, 0.12 * 0.025)
        self.assertAlmostEqual(first.s_max_kva, 3 * 0.127 * 100.0)
        self.assertEqual(len(first.r_matrix_ohm), 3)
        self.assertIn("calcvoltagebases", fake.commands)

    def test_resolves_bus_names_and_integer_ids(self):
        mapping = {"bus_001": 1, "bus_006": 6}
        self.assertEqual(resolve_bus_id("bus_006.1.2.3", mapping), 6)
        self.assertEqual(resolve_bus_id("6", mapping), 6)
        with self.assertRaisesRegex(ValueError, "Unknown OpenDSS bus"):
            resolve_bus_id("missing", mapping)

    def test_load_case_uses_opendss_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            (case_dir / "dss").mkdir()
            (case_dir / "dss" / "Master.dss").write_text("! fake", encoding="utf-8")
            self._write_case_files(case_dir)

            with patch("opf.opendss._new_dss", return_value=FakeDSS()):
                case = load_case(case_dir)

        self.assertEqual(case.bus_name_to_id["bus_002"], 2)
        self.assertEqual(case.grid.bus, 1)
        self.assertEqual(case.bess[0].bus, 2)
        self.assertEqual(case.bess[0].s_max_kva, 2.0)
        self.assertFalse(case.bess[0].reactive_control)
        self.assertEqual(case.bess[0].q_loss_rated_kw, 0.0)
        self.assertEqual(case.pv[0].bus, 3)
        self.assertEqual(case.pv[0].q_loss_rated_kw, 0.0)
        self.assertEqual(case.buses[2].name, "bus_002")
        self.assertEqual(case.buses[2].phases, (1, 2, 3))
        self.assertTrue(math.isclose(case.branches[0].r_ohm, 0.01575))
        self.assertEqual(case.branches[0].length, 0.025)
        self.assertEqual(case.branches[0].length_units, 3)
        model = build_model(case)
        self.assertEqual(model.nobjectives(), 1)
        self.assertEqual(model.qbess["b1", 0].bounds, (0.0, 0.0))
        self.assertEqual(model.pbess_loss["b1", 0].bounds, (0.0, 0.0))
        self.assertEqual(model.ppv_loss["pv1", 0].bounds, (0.0, 0.0))

    def test_real_opendss_fixture(self):
        master = Path(__file__).parent / "fixtures" / "opendss" / "Master.dss"
        caller_directory = Path.cwd()
        network = load_opendss_network(master, slack_bus="bus_001")
        self.assertEqual(Path.cwd(), caller_directory)
        self.assertEqual(len(network.buses), 3)
        self.assertEqual(len(network.branches), 2)
        self.assertAlmostEqual(network.branches[0].r_ohm, 0.63 * 0.025, places=6)

    @staticmethod
    def _write_case_files(path: Path):
        config = {
            "name": "dss-test",
            "base": {"s_base_kva": 100.0, "v_base_kv": 0.22},
            "network": {
                "master": "dss/Master.dss",
                "slack_bus": "bus_001",
            },
            "grid": {
                "bus": "bus_001", "v_ref_pu": 1.0,
                "p_import_max_kw": 100.0, "p_export_max_kw": 100.0,
                "q_max_kvar": 100.0, "feed_in_tariff_ratio": 1.0,
            },
            "voltage_limits": {"v_min_pu": 0.95, "v_max_pu": 1.05},
        }
        devices = {
            "bess": [{
                "id": "b1", "bus": "bus_002", "e_cap_kwh": 10.0,
                "p_charge_max_kw": 2.0, "p_discharge_max_kw": 2.0,
                "eta_charge": 0.95, "eta_discharge": 0.95,
                "soc_init_frac": 0.5,
            }],
            "pv": [{
                "id": "pv1", "bus": "bus_003", "p_max_kw": 2.0,
                "profile": "pv.csv:PV3", "control": "optimal",
            }],
        }
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "devices.json").write_text(json.dumps(devices), encoding="utf-8")
        (path / "demand.csv").write_text(
            "timestamp,Pbus_002,Qbus_002,Pbus_003,Qbus_003\n"
            "2026-01-01 00:00:00,1.0,0.2,2.0,0.4\n"
            "2026-01-01 01:00:00,1.1,0.2,2.1,0.4\n",
            encoding="utf-8",
        )
        (path / "price.csv").write_text(
            "timestamp,price_per_kwh\n"
            "2026-01-01 00:00:00,0.3\n"
            "2026-01-01 01:00:00,0.4\n",
            encoding="utf-8",
        )
        (path / "pv.csv").write_text(
            "timestamp,PV3\n"
            "2026-01-01 00:00:00,0.0\n"
            "2026-01-01 01:00:00,1.0\n",
            encoding="utf-8",
        )


class Case5Test(unittest.TestCase):
    def test_case5_uses_opendss_as_the_static_network_source(self):
        case = load_case("examples/case5")
        model = build_model(case)

        self.assertEqual(case.bus_name_to_id["bus_005"], 5)
        self.assertEqual(len(case.buses), 5)
        self.assertEqual(len(case.branches), 4)
        expected = {
            (1, 2): (1.0, 2.0, 1500.0),
            (2, 3): (2.0, 3.0, 1000.0),
            (3, 4): (2.4, 3.2, 600.0),
            (2, 5): (2.0, 2.6, 800.0),
        }
        for branch in case.branches:
            r, x, s_max = expected[(branch.from_bus, branch.to_bus)]
            self.assertAlmostEqual(branch.r_ohm, r)
            self.assertAlmostEqual(branch.x_ohm, x)
            self.assertAlmostEqual(branch.s_max_kva, s_max)
        bess = case.bess[0]
        self.assertTrue(bess.reactive_control)
        self.assertEqual(bess.e_cap_kwh, 100.0)
        self.assertEqual(bess.s_max_kva, 50.0)
        self.assertEqual(bess.q_loss_rated_kw, 0.5)
        self.assertEqual(case.pv[0].q_loss_rated_kw, 3.0)
        self.assertEqual(model.qbess[bess.id, 0].bounds, (-0.05, 0.05))
        self.assertEqual(model.pbess_loss[bess.id, 0].bounds, (0.0, 0.0005))
        self.assertEqual(model.nvariables(), 672)
        self.assertEqual(model.nconstraints(), 577)

    def test_reactive_losses_follow_rated_quadratic_curve_and_drain_soc(self):
        case = load_case("examples/case5")
        model = build_model(case)

        model.qbess["b1", 0].set_value(0.025)
        model.pbess_loss["b1", 0].set_value(0.000125)
        self.assertAlmostEqual(pyo.value(model.bess_q_loss["b1", 0].body), 0.0)

        model.qpv["pv1", 0].set_value(0.15)
        model.ppv_loss["pv1", 0].set_value(0.00075)
        self.assertAlmostEqual(pyo.value(model.pv_q_loss["pv1", 0].body), 0.0)

        model.pch["b1", 0].set_value(0.0)
        model.pdis["b1", 0].set_value(0.0)
        model.soc["b1", 0].set_value(0.05 - 0.000125)
        self.assertAlmostEqual(pyo.value(model.soc_balance["b1", 0].body), 0.0)


class TransformerCaseTest(unittest.TestCase):
    def test_two_winding_transformer_is_imported_and_model_builds(self):
        case_path = Path(__file__).parent / "fixtures" / "transformer_case"
        case = load_case(case_path)
        model = build_model(case)

        self.assertEqual(len(case.buses), 3)
        self.assertEqual(len(case.branches), 2)
        transformer = case.branches[0]
        line = case.branches[1]
        self.assertEqual(transformer.element_type, "transformer")
        self.assertEqual((transformer.from_bus, transformer.to_bus), (1, 2))
        self.assertAlmostEqual(transformer.r_pu_on_rating, 0.01)
        self.assertAlmostEqual(transformer.x_pu_on_rating, 0.06)
        self.assertAlmostEqual(transformer.tap_ratio, 1.025)
        self.assertEqual(transformer.connections, ("wye", "wye"))
        self.assertAlmostEqual(transformer.r_ohm, 1.21)
        self.assertAlmostEqual(transformer.x_ohm, 7.26)
        self.assertEqual(line.element_type, "line")

        r_tx, x_tx = transformer.impedance_pu(case.base, case.buses[1])
        r_line, x_line = line.impedance_pu(case.base, case.buses[2])
        self.assertAlmostEqual(r_tx, 0.01)
        self.assertAlmostEqual(x_tx, 0.06)
        self.assertAlmostEqual(r_line, 0.01 / 0.16, places=6)
        self.assertAlmostEqual(x_line, 0.02 / 0.16, places=6)
        self.assertEqual(model.nobjectives(), 1)

    def test_transformer_tap_is_inverted_when_branch_orientation_reverses(self):
        master = (Path(__file__).parent / "fixtures" / "transformer_case"
                  / "dss" / "Master.dss")
        network = load_opendss_network(
            master,
            slack_bus="low",
            bus_ids={"source": 1, "low": 2, "load": 3},
        )
        transformer = next(
            branch for branch in network.branches
            if branch.element_type == "transformer"
        )
        self.assertEqual((transformer.from_bus, transformer.to_bus), (2, 1))
        self.assertAlmostEqual(transformer.tap_ratio, 1.0 / 1.025)


class SocpGapTest(unittest.TestCase):
    def test_gap_analysis_reports_tight_and_loose_relaxations(self):
        case_path = Path(__file__).parent / "fixtures" / "transformer_case"
        case = load_case(case_path)
        model = build_model(case)

        for b in case.buses:
            for t in case.periods:
                model.v[b, t].set_value(1.0)
        for j in (bus for bus in case.buses if bus != case.root):
            for t in case.periods:
                model.P[j, t].set_value(0.0)
                model.Q[j, t].set_value(0.0)
                model.l[j, t].set_value(0.0)

        tight = analyze_socp_gap(model, case, tolerance=1e-6)
        self.assertTrue(tight["relaxation_tight"])
        self.assertEqual(tight["tightness"], "tight")
        self.assertGreater(tight["tightness_margin"], 0.0)
        self.assertEqual(tight["max_current_error_a"], 0.0)

        model.l[2, 0].set_value(1e-3)
        loose = analyze_socp_gap(model, case, tolerance=1e-6)
        self.assertFalse(loose["relaxation_tight"])
        self.assertEqual(loose["tightness"], "not_tight")
        self.assertLess(loose["tightness_margin"], 0.0)
        self.assertAlmostEqual(loose["max_normalized"], 1e-3)
        self.assertAlmostEqual(loose["max_relative_flow"], 1.0)
        self.assertGreater(loose["max_current_error_a"], 0.0)
        self.assertGreater(loose["max_loss_error_w"], 0.0)

    def test_gap_tolerance_must_be_positive(self):
        case_path = Path(__file__).parent / "fixtures" / "transformer_case"
        case = load_case(case_path)
        model = build_model(case)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            analyze_socp_gap(model, case, tolerance=0.0)


if __name__ == "__main__":
    unittest.main()
