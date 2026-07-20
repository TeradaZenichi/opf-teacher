"""Conversão da solução Pyomo para as unidades do caso."""
from __future__ import annotations

import math

import pandas as pd
import pyomo.environ as pyo

from opf.components import (
    BessResult, BranchResult, BusResult, Case, GridResult, PvResult, Summary,
)


DEFAULT_SOCP_GAP_TOLERANCE = 1e-6


def _val(x) -> float:
    v = pyo.value(x, exception=False)
    return float(v) if v is not None else float("nan")


def analyze_socp_gap(m, case: Case, tolerance: float = DEFAULT_SOCP_GAP_TOLERANCE):
    """Calcula o resíduo da igualdade relaxada P² + Q² = v*l."""
    if tolerance <= 0.0:
        raise ValueError("SOCP gap tolerance must be positive")

    idx = case.index
    per_branch = {}
    absolute_values = []
    normalized_values = []
    for j in (bus for bus in case.buses if bus != case.root):
        branch = case.branches[case.parent_branch[j]]
        gap_values = []
        normalized = []
        for t in case.periods:
            voltage_current = _val(m.v[branch.from_bus, t]) * _val(m.l[j, t])
            flow_squared = _val(m.P[j, t]) ** 2 + _val(m.Q[j, t]) ** 2
            residual = voltage_current - flow_squared
            scale = max(1.0, abs(voltage_current), abs(flow_squared))
            gap_values.append(residual)
            normalized.append(abs(residual) / scale)
        per_branch[j] = {
            "gap_pu2": pd.Series(gap_values, index=idx),
            "normalized": pd.Series(normalized, index=idx),
        }
        absolute_values.extend(abs(value) for value in gap_values)
        normalized_values.extend(normalized)

    max_absolute = max(absolute_values, default=0.0)
    max_normalized = max(normalized_values, default=0.0)
    margin = tolerance - max_normalized
    if max_normalized <= tolerance * 0.1:
        confidence = "high"
    elif max_normalized <= tolerance:
        confidence = "acceptable"
    else:
        confidence = "low"
    return {
        "branches": per_branch,
        "max_gap_pu2": max_absolute,
        "max_normalized": max_normalized,
        "tolerance": tolerance,
        "confidence_margin": margin,
        "confidence": confidence,
        "relaxation_tight": max_normalized <= tolerance,
    }


def attach_results(
    m,
    case: Case,
    status: str,
    socp_gap_tolerance: float = DEFAULT_SOCP_GAP_TOLERANCE,
) -> Case:
    base = case.base
    idx = case.index
    T = list(case.periods)
    gap_analysis = analyze_socp_gap(m, case, socp_gap_tolerance)

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
        bus.result = BusResult(
            v_pu=series([math.sqrt(max(_val(m.v[b, t]), 0.0)) for t in T])
        )

    for j, branch in ((j, case.branches[case.parent_branch[j]])
                      for j in case.buses if j != case.root):
        r_pu, _ = branch.impedance_pu(base, case.buses[branch.from_bus])
        branch.result = BranchResult(
            p_kw=series([base.to_kw(_val(m.P[j, t])) for t in T]),
            q_kvar=series([base.to_kw(_val(m.Q[j, t])) for t in T]),
            loss_kw=series([base.to_kw(r_pu * _val(m.l[j, t])) for t in T]),
            socp_gap_pu2=gap_analysis["branches"][j]["gap_pu2"],
            socp_gap_normalized=gap_analysis["branches"][j]["normalized"],
        )

    for s in case.bess:
        ch = series([base.to_kw(_val(m.pch[s.id, t])) for t in T])
        dis = series([base.to_kw(_val(m.pdis[s.id, t])) for t in T])
        soc = series([base.to_kwh(_val(m.soc[s.id, t])) for t in T])
        s.result = BessResult(
            charge_kw=ch,
            discharge_kw=dis,
            p_net_kw=ch - dis,
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
        socp_gap_max_pu2=gap_analysis["max_gap_pu2"],
        socp_gap_max_normalized=gap_analysis["max_normalized"],
        socp_gap_tolerance=gap_analysis["tolerance"],
        socp_confidence_margin=gap_analysis["confidence_margin"],
        socp_confidence=gap_analysis["confidence"],
        socp_relaxation_tight=gap_analysis["relaxation_tight"],
    )
    return case
