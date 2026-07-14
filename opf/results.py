"""Pull the Pyomo solution back onto the Case components.

Each component (Grid, Bus, Branch, Bess, Pv) gets a `.result` with time series in
physical units (kW/kVAr/kV/kWh); the Case gets a `.summary`.
"""
from __future__ import annotations

import math

import pandas as pd
import pyomo.environ as pyo

from opf.components import (
    BessResult, BranchResult, BusResult, Case, GridResult, PvResult, Summary,
)


def _val(x) -> float:
    v = pyo.value(x, exception=False)
    return float(v) if v is not None else float("nan")


def attach_results(m, case: Case, status: str) -> Case:
    base = case.base
    idx = case.index
    T = list(case.periods)

    def series(values) -> pd.Series:
        return pd.Series(values, index=idx)

    imp = series([base.to_kw(_val(m.pimp[t])) for t in T])
    exp = series([base.to_kw(_val(m.pexp[t])) for t in T])
    cost = (imp - case.grid.feed_in_ratio * exp) * case.price * case.dt_h
    case.grid.result = GridResult(
        import_kw=imp,
        export_kw=exp,
        q_kvar=series([base.to_kw(_val(m.qgrid[t])) for t in T]),
        cost=cost,
    )

    for b, bus in case.buses.items():
        bus.result = BusResult(  # v is stored squared, so take the root
            v_pu=series([math.sqrt(max(_val(m.v[b, t]), 0.0)) for t in T])
        )

    for j, branch in ((j, case.branches[case.parent_branch[j]])
                      for j in case.buses if j != case.root):
        r_pu = base.pu_impedance(branch.r_ohm)
        branch.result = BranchResult(
            p_kw=series([base.to_kw(_val(m.P[j, t])) for t in T]),
            q_kvar=series([base.to_kw(_val(m.Q[j, t])) for t in T]),
            loss_kw=series([base.to_kw(r_pu * _val(m.l[j, t])) for t in T]),
        )

    for s in case.bess:
        ch = series([base.to_kw(_val(m.pch[s.id, t])) for t in T])
        dis = series([base.to_kw(_val(m.pdis[s.id, t])) for t in T])
        soc = series([base.to_kwh(_val(m.soc[s.id, t])) for t in T])
        s.result = BessResult(
            charge_kw=ch,
            discharge_kw=dis,
            p_net_kw=ch - dis,                 # charge (+) / discharge (-)
            soc_kwh=soc,
            soc_frac=soc / s.e_cap_kwh,
        )

    for g in case.pv:
        gen = series([base.to_kw(_val(m.ppv[g.id, t])) for t in T])
        g.result = PvResult(
            avail_kw=g.avail_kw.copy(),
            gen_kw=gen,
            curtail_kw=g.avail_kw - gen,
            q_kvar=series([base.to_kw(_val(m.qpv[g.id, t])) for t in T]),
        )

    case.summary = Summary(
        status=status,
        objective_cost=_val(m.cost),
        energy_cost=float(cost.sum()),
        energy_import_kwh=float(imp.sum()) * case.dt_h,
        energy_export_kwh=float(exp.sum()) * case.dt_h,
        pv_generated_kwh=sum(float(g.result.gen_kw.sum()) for g in case.pv) * case.dt_h,
        pv_curtailed_kwh=sum(float(g.result.curtail_kw.sum()) for g in case.pv) * case.dt_h,
        losses_kwh=sum(float(case.branches[case.parent_branch[j]].result.loss_kw.sum())
                       for j in case.buses if j != case.root) * case.dt_h,
        v_min_pu=min(float(bus.result.v_pu.min()) for bus in case.buses.values()),
        v_max_pu=max(float(bus.result.v_pu.max()) for bus in case.buses.values()),
    )
    return case
