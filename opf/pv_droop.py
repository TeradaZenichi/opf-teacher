"""Controles locais Volt-VAr e Volt-Watt."""
import pyomo.environ as pyo

VV_V = (0.90, 0.92, 0.98, 1.02, 1.08, 1.15)
VV_Q = (1.0, 1.0, 0.0, 0.0, -1.0, -1.0)
VW_V = (0.90, 1.06, 1.10, 1.15)
VW_P = (1.0, 1.0, 0.0, 0.0)
QMAX_FRAC = 0.44
V_MAX_EMERG = 1.10

MODES = ("volt-var", "volt-watt", "volt-var-watt")


def is_droop(control):
    return control in MODES


def relaxes_voltage(case):
    return V_MAX_EMERG if any("watt" in g.control for g in case.pv) else None


def ppv_bounds(avail):
    return (0.0, avail)


def add(m, pv, pv_ctrl, pv_s, pv_avail, T):
    var_ids = [g for g in pv if "var" in pv_ctrl[g]]
    watt_ids = [g for g in pv if "watt" in pv_ctrl[g]]
    if not (var_ids or watt_ids):
        return

    pure_watt = [(g, t) for g in pv if pv_ctrl[g] == "volt-watt" for t in T]
    if pure_watt:
        m.vw_noq_idx = pyo.Set(initialize=pure_watt, dimen=2)
        m.vw_noq = pyo.Constraint(m.vw_noq_idx, rule=lambda m, g, t: m.qpv[g, t] == 0.0)

    m.vloc = pyo.Var(m.G, m.T)
    m.vloc_link = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.vloc[g, t] == m.v[pv[g].bus, t])

    def _pwl(name, ids, v_pts):
        xk = [V * V for V in v_pts]
        K = list(range(len(xk)))
        S = pyo.Set(initialize=[(g, t) for g in ids for t in T], dimen=2)
        KS = pyo.Set(initialize=K)
        lam = pyo.Var(S, KS, domain=pyo.NonNegativeReals)
        setattr(m, f"{name}_idx", S)
        setattr(m, f"{name}_k", KS)
        setattr(m, f"{name}_lam", lam)
        setattr(m, f"{name}_sum", pyo.Constraint(
            S, rule=lambda m, g, t: sum(lam[g, t, k] for k in K) == 1))
        setattr(m, f"{name}_x", pyo.Constraint(
            S, rule=lambda m, g, t: m.vloc[g, t] == sum(lam[g, t, k] * xk[k] for k in K)))
        setattr(m, f"{name}_sos", pyo.SOSConstraint(
            S, rule=lambda m, g, t: [lam[g, t, k] for k in K], sos=2))
        return S, lam, K

    if var_ids:
        S, lam, K = _pwl("vv", var_ids, VV_V)
        m.volt_var = pyo.Constraint(
            S, rule=lambda m, g, t: m.qpv[g, t]
            == sum(lam[g, t, k] * (QMAX_FRAC * pv_s[g] * VV_Q[k]) for k in K))

    if watt_ids:
        S, lam, K = _pwl("vw", watt_ids, VW_V)
        m.volt_watt = pyo.Constraint(
            S, rule=lambda m, g, t: m.ppv[g, t]
            <= sum(lam[g, t, k] * (VW_P[k] * pv_avail[g][t]) for k in K))
