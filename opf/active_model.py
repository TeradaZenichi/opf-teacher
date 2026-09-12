"""Compatibility imports for the single-phase active-power formulation."""

from opf.single_phase.active_model import (
    DEFAULT_TIE_BREAK_PENALTY,
    _attach_active_results,
    build_active_power_model,
    solve_active_power_opf,
)

__all__ = [
    "DEFAULT_TIE_BREAK_PENALTY",
    "build_active_power_model",
    "solve_active_power_opf",
]
