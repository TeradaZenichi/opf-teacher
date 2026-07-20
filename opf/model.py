"""Modelo DistFlow com relaxação SOCP."""
from __future__ import annotations

import math

import pyomo.environ as pyo

from opf.components import Case
import opf.pv_droop as pv_droop
import opf.pv_optimal as pv_optimal


def build_model(case: Case) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel(name=case.name)
    base = case.base
    dt = case.dt_h
    T = list(case.periods)
    NR = [b for b in case.buses if b != case.root]

    br = {j: case.branches[case.parent_branch[j]] for j in NR}
    RX = {j: br[j].impedance_pu(base, case.buses[br[j].from_bus]) for j in NR}
    R = {j: RX[j][0] for j in NR}
    X = {j: RX[j][1] for j in NR}
    TAP = {j: br[j].tap_ratio for j in NR}
    L_MAX = {j: base.pu_power(br[j].s_max_kva) ** 2 for j in NR}

    pL = {b: base.pu_power(case.buses[b].p_load_kw.to_numpy()) for b in case.buses}
    qL = {b: base.pu_power(case.buses[b].q_load_kw.to_numpy()) for b in case.buses}
    price = case.price.to_numpy()

    bess = {s.id: s for s in case.bess}
    pv = {g.id: g for g in case.pv}
    bess_at = {b: [s for s in case.bess if s.bus == b] for b in case.buses}
    pv_at = {b: [g for g in case.pv if g.bus == b] for b in case.buses}
    pv_avail = {g.id: base.pu_power(g.avail_kw.to_numpy()) for g in case.pv}
    pv_qratio = {g.id: math.tan(math.acos(g.power_factor)) for g in case.pv}
    pv_s = {g.id: base.pu_power(g.s_max_kva) for g in case.pv}
    pv_ctrl = {g.id: g.control for g in case.pv}
    bess_s = {s.id: base.pu_power(s.s_max_kva) for s in case.bess}
    bess_q_loss = {s.id: base.pu_power(s.q_loss_rated_kw) for s in case.bess}
    pv_q_loss = {g.id: base.pu_power(g.q_loss_rated_kw) for g in case.pv}

    m.T = pyo.Set(initialize=T, ordered=True)
    m.B = pyo.Set(initialize=list(case.buses))
    m.J = pyo.Set(initialize=NR, ordered=True)
    m.S = pyo.Set(initialize=list(bess))
    m.G = pyo.Set(initialize=list(pv))

    # Volt-Watt usa o limite emergencial de tensão definido em pv_droop.
    v_relax = pv_droop.relaxes_voltage(case)
    def v_bounds(m, b, t):
        if b == case.root:
            return (case.grid.v_ref_pu ** 2, case.grid.v_ref_pu ** 2)
        v_up = v_relax if v_relax is not None else case.buses[b].v_max_pu
        return (case.buses[b].v_min_pu ** 2, v_up ** 2)
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
    m.qbess = pyo.Var(m.S, m.T, bounds=lambda m, s, t: (
        -bess_s[s] if bess[s].reactive_control else 0.0,
        bess_s[s] if bess[s].reactive_control else 0.0,
    ))
    m.pbess_loss = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals,
                           bounds=lambda m, s, t: (
                               0.0,
                               bess_q_loss[s] if bess[s].reactive_control else 0.0,
                           ))
    m.soc = pyo.Var(m.S, m.T, bounds=lambda m, s, t: (
        bess[s].soc_min_frac * base.pu_energy(bess[s].e_cap_kwh),
        bess[s].soc_max_frac * base.pu_energy(bess[s].e_cap_kwh),
    ))

    def ppv_bounds(m, g, t):
        avail = pv_avail[g][t]
        if pv_droop.is_droop(pv_ctrl[g]):
            return pv_droop.ppv_bounds(avail)
        return (0.0, avail)
    m.ppv = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals, bounds=ppv_bounds)
    m.qpv = pyo.Var(m.G, m.T, bounds=lambda m, g, t: (-pv_s[g], pv_s[g]))
    m.ppv_loss = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals,
                         bounds=lambda m, g, t: (0.0, pv_q_loss[g]))

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
        for s in bess_at[b]:
            expr += m.qbess[s.id, t]
        for g in pv_at[b]:
            expr += m.qpv[g.id, t]
        if b == case.root:
            expr += m.qgrid[t]
        return expr

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
        # A impedância do trafo é referida ao lado de envio; em linhas TAP=1.
        return (m.v[j, t] == TAP[j] ** 2 * (
                m.v[i, t]
                - 2.0 * (R[j] * m.P[j, t] + X[j] * m.Q[j, t])
                + (R[j] ** 2 + X[j] ** 2) * m.l[j, t]))
    m.voltage_drop = pyo.Constraint(m.J, m.T, rule=vdrop)

    def pv_capability(m, g, t):
        return m.ppv[g, t] ** 2 + m.qpv[g, t] ** 2 <= pv_s[g] ** 2
    m.pv_capability = pyo.Constraint(m.G, m.T, rule=pv_capability)

    def pv_available_power(m, g, t):
        used = m.ppv[g, t] + m.ppv_loss[g, t]
        if pv[g].curtailable or pv_droop.is_droop(pv_ctrl[g]):
            return used <= pv_avail[g][t]
        return used == pv_avail[g][t]
    m.pv_available_power = pyo.Constraint(m.G, m.T, rule=pv_available_power)

    def pv_q_loss_rule(m, g, t):
        return pv_q_loss[g] * m.qpv[g, t] ** 2 <= pv_s[g] ** 2 * m.ppv_loss[g, t]
    m.pv_q_loss = pyo.Constraint(m.G, m.T, rule=pv_q_loss_rule)

    def bess_capability(m, s, t):
        p_net = m.pdis[s, t] - m.pch[s, t]
        return p_net ** 2 + m.qbess[s, t] ** 2 <= bess_s[s] ** 2
    m.bess_capability = pyo.Constraint(m.S, m.T, rule=bess_capability)

    def bess_q_loss_rule(m, s, t):
        return (bess_q_loss[s] * m.qbess[s, t] ** 2
                <= bess_s[s] ** 2 * m.pbess_loss[s, t])
    m.bess_q_loss = pyo.Constraint(m.S, m.T, rule=bess_q_loss_rule)

    pv_optimal.add(m, pv, pv_ctrl, pv_qratio, T)
    pv_droop.add(m, pv, pv_ctrl, pv_s, pv_avail, T)

    def cone(m, j, t):
        return m.P[j, t] ** 2 + m.Q[j, t] ** 2 <= m.v[br[j].from_bus, t] * m.l[j, t]
    m.socp = pyo.Constraint(m.J, m.T, rule=cone)

    def soc_rule(m, s, t):
        d = bess[s]
        gain = (d.eta_charge * m.pch[s, t]
                - m.pdis[s, t] / d.eta_discharge
                - m.pbess_loss[s, t]) * dt
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

    # Evita importação e exportação simultâneas sem variável binária.
    def no_roundtrip(m, t):
        return [m.pimp[t], m.pexp[t]]
    m.no_roundtrip = pyo.SOSConstraint(m.T, rule=no_roundtrip, sos=1)

    r_fi = case.grid.feed_in_ratio
    s_base = base.s_base_kva

    def cost(m):
        return s_base * dt * sum(price[t] * (m.pimp[t] - r_fi * m.pexp[t]) for t in T)
    m.cost = pyo.Objective(rule=cost, sense=pyo.minimize)

    return m
