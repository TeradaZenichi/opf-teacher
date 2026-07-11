"""Modelo DistFlow (Baran-Wu) com relaxacao conica (SOCP) em Pyomo.

Le os parametros fisicos do Case e os converte para por-unidade (via case.base)
ao montar o modelo. Variaveis:
    v[b,t] = |V_b|^2   (tensao ao quadrado, p.u.)
    l[j,t] = |I_j|^2   (corrente ao quadrado no ramo que entra na barra j)
    P[j,t], Q[j,t]     (fluxos no extremo emissor do ramo i->j)

Relaxacao conica:  P^2 + Q^2 <= v_i * l . Por padrao usa-se a aproximacao de
tensao fixa v_i ~ 1 p.u. -> P^2 + Q^2 <= l ; exact_cone=True usa o cone rotacionado.
"""
from __future__ import annotations

import math

import pyomo.environ as pyo

from opf.components import Case


def build_model(case: Case, *, exact_cone: bool = False) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel(name=case.name)
    base = case.base
    dt = case.dt_h
    T = list(case.periods)
    NR = [b for b in case.buses if b != case.root]            # barras nao-raiz == ramos

    # ramo de entrada de cada barra filha + parametros em p.u.
    br = {j: case.branches[case.parent_branch[j]] for j in NR}
    R = {j: base.pu_impedance(br[j].r_ohm) for j in NR}
    X = {j: base.pu_impedance(br[j].x_ohm) for j in NR}
    L_MAX = {j: base.pu_power(br[j].s_max_kva) ** 2 for j in NR}

    # cargas em p.u. (arrays por barra, indexados por periodo)
    pL = {b: base.pu_power(case.buses[b].p_load_kw.to_numpy()) for b in case.buses}
    qL = {b: base.pu_power(case.buses[b].q_load_kw.to_numpy()) for b in case.buses}
    price = case.price.to_numpy()

    # dispositivos por barra + dicionarios por id
    bess = {s.id: s for s in case.bess}
    pv = {g.id: g for g in case.pv}
    bess_at = {b: [s for s in case.bess if s.bus == b] for b in case.buses}
    pv_at = {b: [g for g in case.pv if g.bus == b] for b in case.buses}
    pv_avail = {g.id: base.pu_power(g.avail_kw.to_numpy()) for g in case.pv}
    pv_qratio = {g.id: math.tan(math.acos(g.power_factor)) for g in case.pv}

    # ------------------------------------------------------------------ sets
    m.T = pyo.Set(initialize=T, ordered=True)
    m.B = pyo.Set(initialize=list(case.buses))
    m.J = pyo.Set(initialize=NR, ordered=True)
    m.S = pyo.Set(initialize=list(bess))
    m.G = pyo.Set(initialize=list(pv))

    # -------------------------------------------------------------- variaveis
    def v_bounds(m, b, t):
        if b == case.root:
            return (case.grid.v_ref_pu ** 2, case.grid.v_ref_pu ** 2)
        return (case.buses[b].v_min_pu ** 2, case.buses[b].v_max_pu ** 2)
    m.v = pyo.Var(m.B, m.T, bounds=v_bounds)

    m.P = pyo.Var(m.J, m.T)
    m.Q = pyo.Var(m.J, m.T)
    m.l = pyo.Var(m.J, m.T, domain=pyo.NonNegativeReals,
                  bounds=lambda m, j, t: (0.0, L_MAX[j]))

    m.pimp = pyo.Var(m.T, bounds=(0.0, base.pu_power(case.grid.p_import_max_kw)))
    m.pexp = pyo.Var(m.T, bounds=(0.0, base.pu_power(case.grid.p_export_max_kw)))
    qg = base.pu_power(case.grid.q_max_kvar)
    m.qgrid = pyo.Var(m.T, bounds=(-qg, qg))

    m.pch = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals,
                    bounds=lambda m, s, t: (0.0, base.pu_power(bess[s].p_charge_max_kw)))
    m.pdis = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals,
                     bounds=lambda m, s, t: (0.0, base.pu_power(bess[s].p_discharge_max_kw)))
    m.soc = pyo.Var(m.S, m.T, bounds=lambda m, s, t: (
        bess[s].soc_min_frac * base.pu_energy(bess[s].e_cap_kwh),
        bess[s].soc_max_frac * base.pu_energy(bess[s].e_cap_kwh),
    ))

    def ppv_bounds(m, g, t):
        avail = pv_avail[g][t]
        return (0.0, avail) if pv[g].curtailable else (avail, avail)
    m.ppv = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals, bounds=ppv_bounds)

    # ---------------------------------------------------- injecoes liquidas
    def inj_p(m, b, t):
        expr = -pL[b][t]
        for s in bess_at[b]:
            expr += m.pdis[s.id, t] - m.pch[s.id, t]
        for g in pv_at[b]:
            expr += m.ppv[g.id, t]
        if b == case.root:
            expr += m.pimp[t] - m.pexp[t]
        return expr

    def inj_q(m, b, t):
        expr = -qL[b][t]
        for g in pv_at[b]:
            expr += pv_qratio[g.id] * m.ppv[g.id, t]
        if b == case.root:
            expr += m.qgrid[t]
        return expr

    # --------------------------------------------------------- restricoes
    def bal_p(m, b, t):
        out = sum(m.P[k, t] for k in case.children.get(b, []))
        incoming = (m.P[b, t] - R[b] * m.l[b, t]) if b != case.root else 0.0
        return incoming + inj_p(m, b, t) == out
    m.balance_p = pyo.Constraint(m.B, m.T, rule=bal_p)

    def bal_q(m, b, t):
        out = sum(m.Q[k, t] for k in case.children.get(b, []))
        incoming = (m.Q[b, t] - X[b] * m.l[b, t]) if b != case.root else 0.0
        return incoming + inj_q(m, b, t) == out
    m.balance_q = pyo.Constraint(m.B, m.T, rule=bal_q)

    def vdrop(m, j, t):
        i = br[j].from_bus
        return (m.v[j, t] == m.v[i, t]
                - 2.0 * (R[j] * m.P[j, t] + X[j] * m.Q[j, t])
                + (R[j] ** 2 + X[j] ** 2) * m.l[j, t])
    m.voltage_drop = pyo.Constraint(m.J, m.T, rule=vdrop)

    def cone(m, j, t):
        rhs = m.v[br[j].from_bus, t] * m.l[j, t] if exact_cone else m.l[j, t]
        return m.P[j, t] ** 2 + m.Q[j, t] ** 2 <= rhs
    m.socp = pyo.Constraint(m.J, m.T, rule=cone)

    def soc_rule(m, s, t):
        d = bess[s]
        gain = (d.eta_charge * m.pch[s, t] - m.pdis[s, t] / d.eta_discharge) * dt
        soc0 = d.soc_init_frac * base.pu_energy(d.e_cap_kwh)
        prev = soc0 if t == T[0] else m.soc[s, T[T.index(t) - 1]]
        return m.soc[s, t] == prev + gain
    m.soc_balance = pyo.Constraint(m.S, m.T, rule=soc_rule)

    def soc_cyclic(m, s):
        d = bess[s]
        if not d.cyclic_soc:
            return pyo.Constraint.Skip
        return m.soc[s, T[-1]] == d.soc_init_frac * base.pu_energy(d.e_cap_kwh)
    m.soc_terminal = pyo.Constraint(m.S, rule=soc_cyclic)

    # nao-simultaneidade import/export: SOS1 garante que no maximo um de
    # {pimp, pexp} e nao-nulo por periodo, evitando round-trip degenerado sem
    # poluir o objetivo e sem variaveis binarias (o solver ramifica -> MISOCP).
    def no_roundtrip(m, t):
        return [m.pimp[t], m.pexp[t]]
    m.no_roundtrip = pyo.SOSConstraint(m.T, rule=no_roundtrip, sos=1)

    # --------------------------------------------------------------- objetivo
    r_fi = case.grid.feed_in_ratio
    s_base = base.s_base_kva

    def cost(m):
        return s_base * dt * sum(price[t] * (m.pimp[t] - r_fi * m.pexp[t]) for t in T)
    m.cost = pyo.Objective(rule=cost, sense=pyo.minimize)

    return m
