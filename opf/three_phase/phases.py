"""Phase identifiers and validation shared by three-phase contracts."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math


PHASES = ("a", "b", "c")
_DSS_PHASES = {1: "a", 2: "b", 3: "c"}


def normalize_phase(value: str | int) -> str:
    """Return the public phase name for an OpenDSS node or phase label."""
    if isinstance(value, bool):
        raise ValueError(f"Invalid phase {value!r}")
    if isinstance(value, int):
        try:
            return _DSS_PHASES[value]
        except KeyError as exc:
            raise ValueError(f"Invalid OpenDSS phase node {value!r}") from exc

    normalized = str(value).strip().lower()
    aliases = {
        "1": "a", "phase_1": "a", "phase1": "a",
        "2": "b", "phase_2": "b", "phase2": "b",
        "3": "c", "phase_3": "c", "phase3": "c",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PHASES:
        raise ValueError(f"Invalid phase {value!r}; expected one of {PHASES}")
    return normalized


def normalize_phases(values: Iterable[str | int]) -> tuple[str, ...]:
    phases = tuple(normalize_phase(value) for value in values)
    if not phases:
        raise ValueError("At least one phase is required")
    if len(set(phases)) != len(phases):
        raise ValueError(f"Duplicate phases are not allowed: {phases}")
    return phases


def phase_values(
    values: Mapping[str | int, float],
    phases: Iterable[str | int],
    *,
    label: str,
) -> dict[str, float]:
    """Normalize a finite scalar mapping and require exactly the given phases."""
    expected = normalize_phases(phases)
    normalized = {normalize_phase(key): float(value) for key, value in values.items()}
    if len(normalized) != len(values):
        raise ValueError(f"{label} contains duplicate phase aliases")
    if set(normalized) != set(expected):
        raise ValueError(
            f"{label} phases must be exactly {expected}; got {tuple(normalized)}"
        )
    if not all(math.isfinite(value) for value in normalized.values()):
        raise ValueError(f"{label} values must be finite")
    return {phase: normalized[phase] for phase in expected}
