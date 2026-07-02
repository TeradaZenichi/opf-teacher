"""Facade de alto nivel para a operacao otima de BESS/DERs em distribuicao.

    from teacher import BessOpt

    case = BessOpt("examples/case5").build().solve()

    print(case.summary)
    print(case.bess[0].result.soc_kwh)     # SoC (kWh) por timestamp
    print(case.grid.result.import_kw)      # importacao da rede (kW)
    print(case.buses[2].result.v_pu)       # tensao (p.u.) na barra 2
"""
from __future__ import annotations

from pathlib import Path

import pyomo.environ as pyo

from opf.components import Case
from opf.data import load_case
from opf.model import build_model
from opf.results import attach_results


class BessOpt:
    def __init__(self, source: str | Path | Case, *, exact_cone: bool = False):
        self.case: Case = source if isinstance(source, Case) else load_case(source)
        self.exact_cone = exact_cone
        self.model: pyo.ConcreteModel | None = None

    def build(self) -> "BessOpt":
        self.model = build_model(self.case, exact_cone=self.exact_cone)
        return self

    def solve(self, solver: str = "gurobi_direct", tee: bool = False, **options) -> Case:
        if self.model is None:
            self.build()
        opt = pyo.SolverFactory(solver)
        if opt is None or not opt.available(exception_flag=False):
            raise RuntimeError(
                f"Solver '{solver}' indisponivel. Para este QCQP convexo use, p.ex., "
                f"gurobi_direct, ipopt ou scip (ver README)."
            )
        for k, v in options.items():
            opt.options[k] = v
        res = opt.solve(self.model, tee=tee)
        return attach_results(self.model, self.case, str(res.solver.termination_condition))
