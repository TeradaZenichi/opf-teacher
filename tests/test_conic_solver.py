import math
import unittest
from copy import deepcopy

import pyomo.environ as pyo

from opf_teacher.opf.single_phase.conic import ConicModel
from opf_teacher.opf.single_phase import load_case
from opf_teacher.opf.single_phase.distflow_socp import build_model
from opf_teacher.opf.single_phase.quality import analyze_quality
from opf_teacher.opf.single_phase.results import analyze_socp_gap
from opf_teacher.teacher import SolveConfig, Teacher


class ConicSolverTest(unittest.TestCase):
    def test_quadratic_epigraph(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(0, None))
        model.y = pyo.Var(bounds=(0.5, 0.5))
        model.loss = pyo.Constraint(expr=model.y ** 2 <= model.x)
        model.cost = pyo.Objective(expr=model.x)
        ConicModel(model).solve(SolveConfig())
        if not math.isclose(pyo.value(model.x), 0.25, abs_tol=1e-6):
            raise ValueError("Incorrect quadratic epigraph solution")

    def test_rotated_power_cone(self):
        model = pyo.ConcreteModel()
        model.p = pyo.Var(bounds=(3, 3))
        model.q = pyo.Var(bounds=(4, 4))
        model.v = pyo.Var(bounds=(2, 2))
        model.current = pyo.Var(bounds=(0, None))
        model.cone = pyo.Constraint(expr=model.p ** 2 + model.q ** 2 <= model.v * model.current)
        model.cost = pyo.Objective(expr=model.current)
        ConicModel(model).solve(SolveConfig())
        if not math.isclose(pyo.value(model.current), 12.5, abs_tol=1e-5):
            raise ValueError("Incorrect rotated-cone solution")

    def test_positive_semidefinite_cross_terms(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(bounds=(-5, 5))
        model.y = pyo.Var(bounds=(0.5, 0.5))
        model.q = pyo.Var(bounds=(0.6, 0.6))
        model.capability = pyo.Constraint(expr=(model.x - model.y) ** 2 + model.q ** 2 <= 1)
        model.cost = pyo.Objective(expr=model.x)
        ConicModel(model).solve(SolveConfig())
        if not math.isclose(pyo.value(model.x), -0.3, abs_tol=1e-6):
            raise ValueError("Incorrect capability-circle solution")

    def test_gap_quality_is_invariant_to_the_power_base(self):
        case = load_case("examples/case5").horizon(0, periods=1)
        other = deepcopy(case)
        other.base.s_base_kva /= 10
        reports = []
        for source, factor in ((case, 1), (other, 10)):
            model = build_model(source)
            for bus in model.B:
                model.v[bus, 0].set_value(1.0)
            for bus in model.J:
                model.P[bus, 0].set_value(0.01 * factor)
                model.Q[bus, 0].set_value(0.005 * factor)
                model.l[bus, 0].set_value(0.00013 * factor ** 2)
            reports.append(analyze_socp_gap(model, source))
        for key in ("max_normalized", "max_relative_flow", "max_current_error_a", "max_loss_error_w"):
            if not math.isclose(reports[0][key], reports[1][key], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Power-base-dependent quality metric: {key}")
        if reports[0]["tightness"] != reports[1]["tightness"]:
            raise ValueError("Power-base-dependent SOCP label")

    def test_loss_underestimate_is_rejected_for_imitation(self):
        teacher = Teacher("examples/case5")
        solved = teacher.solve()
        battery = solved.bess[0]
        reactive = 0.2 * battery.s_max_kva
        teacher.model.qbess[battery.id, 0].set_value(solved.base.pu_power(reactive))
        teacher.model.pbess_loss[battery.id, 0].set_value(0.0)
        battery.result.q_kvar.iloc[0] = reactive
        battery.result.inverter_loss_kw.iloc[0] = 0.0
        quality = analyze_quality(teacher.model, solved)
        if quality["feasible"] or quality["valid_for_imitation"]:
            raise ValueError("An underestimated physical loss was accepted")
        expected = battery.q_loss_rated_kw * 0.2 ** 2 * 1000
        if not math.isclose(quality["loss_underestimate_max_w"], expected, abs_tol=1e-8):
            raise ValueError("Incorrect physical loss violation")


if __name__ == "__main__":
    unittest.main()
