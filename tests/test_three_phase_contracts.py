import unittest

import pandas as pd

from opf.components import Base, Pv
from opf.three_phase import (
    BessAction,
    Branch,
    Bus,
    BusState,
    Case,
    ConnectedPv,
    DeviceConnection,
    Grid,
    normalize_phase,
)
from teacher import Teacher, ThreePhaseTeacher


class ThreePhaseContractTest(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2026-01-01", periods=2, freq="h")

    def profile(self, first, second=None):
        return pd.Series(
            [first, first if second is None else second],
            index=self.index,
            dtype=float,
        )

    def bus(self, bus_id, *, asymmetric=False):
        active = {
            "a": self.profile(1.0),
            "b": self.profile(2.0 if asymmetric else 1.0),
            "c": self.profile(3.0 if asymmetric else 1.0),
        }
        reactive = {phase: values * 0.2 for phase, values in active.items()}
        return Bus(
            id=bus_id,
            name=f"bus_{bus_id:03d}",
            phases=(1, 2, 3),
            p_load_kw=active,
            q_load_kvar=reactive,
            kv_base_ln=0.127,
        )

    def valid_case(self):
        buses = {1: self.bus(1), 2: self.bus(2, asymmetric=True)}
        impedance = (
            (0.010 + 0.020j, 0.002 + 0.004j, 0.002 + 0.004j),
            (0.002 + 0.004j, 0.010 + 0.020j, 0.002 + 0.004j),
            (0.002 + 0.004j, 0.002 + 0.004j, 0.010 + 0.020j),
        )
        branch = Branch(
            name="line_001_002",
            from_bus=1,
            to_bus=2,
            phases=("a", "b", "c"),
            z_matrix_ohm=impedance,
            norm_amps={"a": 100.0, "b": 90.0, "c": 80.0},
        )
        grid = Grid(
            bus=1,
            phases=("a", "b", "c"),
            v_ref_pu={"a": 1.0, "b": 1.0, "c": 1.0},
            angle_ref_deg={"a": 0.0, "b": -120.0, "c": 120.0},
            p_import_max_kw=100.0,
            p_export_max_kw=100.0,
            q_max_kvar=100.0,
            feed_in_ratio=0.5,
        )
        return Case(
            name="unbalanced-two-bus",
            base=Base(100.0, 0.22),
            buses=buses,
            branches=[branch],
            grid=grid,
            bess=[],
            pv=[],
            timestamps=self.index,
            dt_h=1.0,
            price=pd.Series([0.5, 0.6], index=self.index),
        )

    def test_phase_contract_keeps_asymmetry_and_mutual_impedance(self):
        case = self.valid_case()

        self.assertEqual(case.buses[2].phases, ("a", "b", "c"))
        self.assertEqual(case.buses[2].p_load_total_kw.iloc[0], 6.0)
        self.assertEqual(case.branches[0].z_matrix_ohm[0][1], 0.002 + 0.004j)
        self.assertEqual(case.branches[0].norm_amps["c"], 80.0)
        self.assertEqual(case.grid.angle_ref_deg["b"], -120.0)

    def test_three_phase_teacher_solves_fixed_dispatch_case(self):
        teacher = Teacher(self.valid_case(), formulation="three_phase_ivr")
        solved = teacher.solve()

        self.assertEqual(solved.summary.status, "locally_optimal")
        self.assertLess(solved.quality["max_power_flow_residual_a"], 1e-6)
        self.assertLess(solved.buses[2].result.v_pu["c"].iloc[0], 1.0)

    def test_three_phase_teacher_uses_phase_native_pre_action_observation(self):
        case = self.valid_case()
        teacher = ThreePhaseTeacher(case)
        observation = {
            "observation_schema_version": 1,
            "timestamp": case.index[0].isoformat(),
            "dt_h": case.dt_h,
            "phase_order": ["a", "b", "c"],
            "buses": {
                bus.name: {
                    "v_before_pu": {phase: 1.0 for phase in bus.phases},
                    "angle_before_deg": {"a": 0.0, "b": -120.0, "c": 120.0},
                    "p_load_kw": {
                        phase: float(bus.p_load_kw[phase].iloc[0])
                        for phase in bus.phases
                    },
                    "q_load_kvar": {
                        phase: float(bus.q_load_kvar[phase].iloc[0])
                        for phase in bus.phases
                    },
                }
                for bus in case.buses.values()
            },
            "grid": {
                "buy_price_per_kwh": 0.5,
                "sell_price_per_kwh": 0.25,
            },
            "bess": {},
            "pv": {},
        }

        solved = teacher.observe_dict(observation).solve()
        x, y = solved.state_action()

        self.assertEqual(x["timestamp"], self.index[0].isoformat())
        self.assertEqual(x["phase_order"], ["a", "b", "c"])
        self.assertEqual(x["buses"][2]["p_load_kw"]["c"], 3.0)
        self.assertEqual(y, {"bess": {}, "pv": {}})

    def test_three_phase_solver_enforces_grid_limits(self):
        case = self.valid_case()
        case.grid.p_import_max_kw = 1.0

        with self.assertRaisesRegex(RuntimeError, "violates network limits"):
            Teacher(case, formulation="three_phase_ivr").solve()

    def test_states_and_actions_are_phase_indexed(self):
        state = BusState(
            v_before_pu={1: 1.01, 2: 0.99, 3: 1.0},
            angle_before_deg={1: 0.0, 2: -120.0, 3: 120.0},
            p_load_kw={1: 1.0, 2: 2.0, 3: 3.0},
            q_load_kvar={1: 0.1, 2: 0.2, 3: 0.3},
        )
        action = BessAction(
            p_net_kw={"a": 1.0, "b": -2.0, "c": 0.5},
            q_injection_kvar={"a": 0.1, "b": 0.2, "c": -0.1},
        )

        self.assertEqual(state.phases, ("a", "b", "c"))
        self.assertEqual(action.p_net_total_kw, -0.5)
        self.assertAlmostEqual(action.q_injection_total_kvar, 0.2)

    def test_device_connection_is_explicit(self):
        single_phase = DeviceConnection(bus=2, phases=(2,), connection="wye")
        delta = DeviceConnection(
            bus=2,
            phases=("a", "b", "c"),
            connection="delta",
            dispatch_mode="per_phase",
        )

        self.assertEqual(single_phase.phases, ("b",))
        self.assertEqual(delta.dispatch_mode, "per_phase")
        with self.assertRaisesRegex(ValueError, "at least two phases"):
            DeviceConnection(bus=2, phases=("a",), connection="delta")

    def test_pv_phase_availability_must_match_aggregate_profile(self):
        device = Pv(
            id="pv1",
            bus=2,
            p_max_kw=10.0,
            s_max_kva=10.0,
            control="optimal",
            curtailable=True,
            power_factor=1.0,
            avail_kw=self.profile(6.0),
        )
        connection = DeviceConnection(bus=2, phases=("a", "b", "c"))
        connected = ConnectedPv(
            device=device,
            connection=connection,
            available_kw={
                "a": self.profile(1.0),
                "b": self.profile(2.0),
                "c": self.profile(3.0),
            },
        )

        self.assertEqual(connected.available_total_kw.iloc[0], 6.0)
        with self.assertRaisesRegex(ValueError, "must sum"):
            ConnectedPv(
                device=device,
                connection=connection,
                available_kw={phase: self.profile(1.0) for phase in ("a", "b", "c")},
            )

    def test_contract_rejects_inconsistent_phase_data(self):
        with self.assertRaisesRegex(ValueError, "2x2"):
            Branch(
                name="bad",
                from_bus=1,
                to_bus=2,
                phases=("a", "b"),
                z_matrix_ohm=((1.0 + 1.0j,),),
                norm_amps=100.0,
            )

        shifted = pd.date_range("2026-01-02", periods=2, freq="h")
        with self.assertRaisesRegex(ValueError, "same index"):
            Bus(
                id=2,
                name="bad_bus",
                phases=("a", "b"),
                p_load_kw={
                    "a": self.profile(1.0),
                    "b": pd.Series([1.0, 1.0], index=shifted),
                },
                q_load_kvar={"a": self.profile(0.1), "b": self.profile(0.1)},
            )

    def test_phase_names_are_strict(self):
        self.assertEqual(normalize_phase(1), "a")
        with self.assertRaisesRegex(ValueError, "Invalid phase"):
            normalize_phase("neutral")


if __name__ == "__main__":
    unittest.main()
