"""Names and compatibility rules for OPF formulations."""
from __future__ import annotations

from enum import Enum


class Formulation(str, Enum):
    """Formulations exposed by :class:`teacher.Teacher`."""

    SINGLE_PHASE_ACTIVE = "single_phase_active"
    SINGLE_PHASE_SOCP = "single_phase_socp"
    THREE_PHASE_IVR = "three_phase_ivr"


_ALIASES = {
    "active_power_lindistflow": Formulation.SINGLE_PHASE_ACTIVE,
    "balanced_active": Formulation.SINGLE_PHASE_ACTIVE,
    "balanced_socp": Formulation.SINGLE_PHASE_SOCP,
    "distflow_socp": Formulation.SINGLE_PHASE_SOCP,
    "unbalanced_ac_ivr": Formulation.THREE_PHASE_IVR,
}


def parse_formulation(value: str | Formulation) -> Formulation:
    if isinstance(value, Formulation):
        return value
    normalized = str(value).strip().lower().replace("-", "_")
    try:
        return Formulation(normalized)
    except ValueError:
        try:
            return _ALIASES[normalized]
        except KeyError as exc:
            supported = ", ".join(item.value for item in Formulation)
            raise ValueError(
                f"Unknown OPF formulation {value!r}; expected one of: {supported}"
            ) from exc


def resolve_formulation(
    formulation: str | Formulation | None,
    *,
    active_power_only: bool,
) -> Formulation:
    """Resolve the new explicit selector and the legacy boolean flag."""
    if formulation is None:
        return (
            Formulation.SINGLE_PHASE_ACTIVE
            if active_power_only
            else Formulation.SINGLE_PHASE_SOCP
        )

    resolved = parse_formulation(formulation)
    if active_power_only and resolved is not Formulation.SINGLE_PHASE_ACTIVE:
        raise ValueError(
            "active_power_only=True conflicts with formulation="
            f"{resolved.value!r}"
        )
    return resolved
