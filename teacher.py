"""Interface de construção e solução do OPF."""
from __future__ import annotations

from pathlib import Path

import pyomo.environ as pyo

from opf.components import Case
from opf.data import load_case
from opf.model import build_model
from opf.results import DEFAULT_SOCP_GAP_TOLERANCE, attach_results


class BessOpt:
    def __init__(self, source: str | Path | Case):
        self.case: Case = source if isinstance(source, Case) else load_case(source)
        self.model: pyo.ConcreteModel | None = None

    def build(self) -> "BessOpt":
        self.model = build_model(self.case)
        return self

    def solve(self, solver: str = "gurobi_direct", tee: bool = False,
              socp_gap_tolerance: float = DEFAULT_SOCP_GAP_TOLERANCE,
              **options) -> Case:
        if self.model is None:
            self.build()
        opt = pyo.SolverFactory(solver)
        if opt is None or not opt.available(exception_flag=False):
            raise RuntimeError(
                f"Solver '{solver}' unavailable; use gurobi_direct or scip."
            )
        for k, v in options.items():
            opt.options[k] = v
        res = opt.solve(self.model, tee=tee)
        return attach_results(
            self.model,
            self.case,
            str(res.solver.termination_condition),
            socp_gap_tolerance=socp_gap_tolerance,
        )
