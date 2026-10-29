"""Supported OPF formulation names."""
from enum import Enum


class Formulation(str, Enum):
    SINGLE_PHASE_ACTIVE = "single_phase_active"
    SINGLE_PHASE_SOCP = "single_phase_socp"
    THREE_PHASE_IVR = "three_phase_ivr"


def parse_formulation(value: str | Formulation) -> Formulation:
    try:
        return Formulation(value)
    except ValueError as error:
        supported = ", ".join(item.value for item in Formulation)
        raise ValueError(f"Unknown OPF formulation {value!r}; expected one of: {supported}") from error
