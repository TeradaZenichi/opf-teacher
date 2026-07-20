"""Controles optimal e fixed_pf do inversor fotovoltaico."""
import pyomo.environ as pyo

MODES = ("optimal", "fixed_pf")


def is_optimal(control):
    return control in MODES


def ppv_bounds(avail, curtailable):
    return (0.0, avail) if curtailable else (avail, avail)


def add(m, pv, pv_ctrl, pv_qratio, T):
    fixed = [(g, t) for g in pv if pv_ctrl[g] == "fixed_pf" for t in T]
    if fixed:
        m.pv_fixed_pf_idx = pyo.Set(initialize=fixed, dimen=2)
        m.pv_fixed_pf = pyo.Constraint(
            m.pv_fixed_pf_idx,
            rule=lambda m, g, t: m.qpv[g, t] == pv_qratio[g] * m.ppv[g, t])
