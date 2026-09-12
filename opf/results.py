"""Compatibility imports for single-phase result processing."""

from opf.single_phase.results import (
    DEFAULT_SOCP_GAP_TOLERANCE,
    RELATIVE_FLOW_FLOOR_PU2,
    analyze_socp_gap,
    attach_results,
)

__all__ = [
    "DEFAULT_SOCP_GAP_TOLERANCE",
    "RELATIVE_FLOW_FLOOR_PU2",
    "analyze_socp_gap",
    "attach_results",
]
