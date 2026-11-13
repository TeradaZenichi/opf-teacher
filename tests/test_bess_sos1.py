from pathlib import Path
from types import SimpleNamespace
import unittest

import pyomo.environ as pyo

from opf_teacher import Teacher
from opf_teacher.opf.single_phase.distflow_socp import build_model
from opf_teacher.opf.single_phase.quality import analyze_quality
from opf_teacher.teacher import SolveConfig


SOURCE = Path(__file__).resolve().parent / "fixtures/single_phase_case"


class BessSos1Test(unittest.TestCase):
    def test_each_battery_period_has_its_own_charge_discharge_sos1(self):
        case = Teacher(SOURCE).case
        model = build_model(case)
        if len(model.bess_no_simultaneous) != len(case.bess) * case.n_periods:
            raise ValueError("Each battery and period requires a separate SOS1")
        for (battery, time), constraint in model.bess_no_simultaneous.items():
            variables = list(constraint.get_variables())
            if constraint.level != 1 or len(variables) != 2:
                raise ValueError("Battery exclusivity must be an SOS1 pair")
            if variables[0] is not model.pch[battery, time] or variables[1] is not model.pdis[battery, time]:
                raise ValueError("SOS1 must connect the same battery's charge and discharge")

    def test_objective_contains_only_grid_energy_cost(self):
        case = Teacher(SOURCE).case
        model = build_model(case)
        for time in model.T:
            model.pgrid[time].set_value(0.02)
            model.cost_rate[time].set_value(float(case.price.iloc[time]) * 0.02)
        expected = case.base.s_base_kva * case.dt_h * float(case.price.sum()) * 0.02
        for variable in (model.pch, model.pdis, model.pbess_loss):
            for value in variable.values():
                value.set_value(value.ub / 2)
        if abs(pyo.value(model.cost) - expected) > 1e-12:
            raise ValueError("Battery powers or losses must not add an objective penalty")

    def test_no_battery_keeps_the_continuous_solver_path(self):
        case = Teacher(SOURCE).case
        case.bess = []
        model = build_model(case)
        if any(model.component_data_objects(pyo.SOSConstraint, active=True)):
            raise ValueError("A case without BESS must not acquire battery SOS constraints")

    def test_quality_rejects_sos1_violation_even_without_energy_equations(self):
        model = pyo.ConcreteModel()
        model.charge = pyo.Var(bounds=(0, 2), initialize=1)
        model.discharge = pyo.Var(bounds=(0, 2), initialize=1)
        model.exclusive = pyo.SOSConstraint(rule=lambda m: [m.charge, m.discharge], sos=1)
        model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        case = SimpleNamespace(bess=[], pv=[], summary=SimpleNamespace(socp_relaxation_tight=True))
        quality = analyze_quality(model, case)
        if quality["feasible"] or quality["valid_for_imitation"] or quality["worst_constraint"] != "exclusive":
            raise ValueError("SOS1 violations must fail quality checks")

    @unittest.skipUnless(pyo.SolverFactory("gurobi_direct").available(exception_flag=False), "Gurobi license required")
    def test_gurobi_removes_simultaneity_and_keeps_loss_checks(self):
        teacher = Teacher(SOURCE)
        teacher.case = teacher.case.horizon(0, 2)
        case = teacher.solve(config=SolveConfig(solver_options={"OutputFlag": 0}))
        if case.quality["solver"]["name"] != "gurobi_direct":
            raise ValueError("Battery SOS1 must select Gurobi by default")
        if case.quality["simultaneous_charge_discharge_max_kw"] > 1e-5:
            raise ValueError("The solved trajectory still charges and discharges simultaneously")
        error = case.quality["energy_execution_error_max_wh"]
        tolerance = case.quality["tolerance"] * case.bess[0].e_cap_kwh * 1000
        if error > tolerance and case.quality["valid_for_imitation"]:
            raise ValueError("Residual inverter-loss inconsistency must not be accepted")
