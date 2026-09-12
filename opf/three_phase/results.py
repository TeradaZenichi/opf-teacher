"""Phase-indexed result contracts for the unbalanced teacher."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import pandas as pd

from opf.three_phase.components import _profile_map
from opf.three_phase.phases import normalize_phases


def _aligned_profiles(values, phases, *, label, index=None):
    profiles = _profile_map(values, phases, label=label)
    profile_index = next(iter(profiles.values())).index
    if index is not None and not profile_index.equals(index):
        raise ValueError(f"{label} does not match the expected result horizon")
    return profiles, profile_index


@dataclass
class BusResult:
    v_pu: Mapping[str | int, pd.Series]
    angle_deg: Mapping[str | int, pd.Series]

    def __post_init__(self):
        phases = normalize_phases(self.v_pu)
        self.v_pu, index = _aligned_profiles(
            self.v_pu, phases, label="bus voltage result"
        )
        self.angle_deg, _ = _aligned_profiles(
            self.angle_deg, phases, label="bus voltage angle result", index=index
        )
        if any((series <= 0.0).any() for series in self.v_pu.values()):
            raise ValueError("Bus voltage result must be positive")


@dataclass
class BranchResult:
    p_kw: Mapping[str | int, pd.Series]
    q_kvar: Mapping[str | int, pd.Series]
    current_a: Mapping[str | int, pd.Series]
    loss_kw: Mapping[str | int, pd.Series]

    def __post_init__(self):
        phases = normalize_phases(self.p_kw)
        self.p_kw, index = _aligned_profiles(
            self.p_kw, phases, label="branch active flow result"
        )
        self.q_kvar, _ = _aligned_profiles(
            self.q_kvar, phases, label="branch reactive flow result", index=index
        )
        self.current_a, _ = _aligned_profiles(
            self.current_a, phases, label="branch current result", index=index
        )
        self.loss_kw, _ = _aligned_profiles(
            self.loss_kw, phases, label="branch loss result", index=index
        )
        if any((series < 0.0).any() for series in self.current_a.values()):
            raise ValueError("Branch current magnitude cannot be negative")


@dataclass
class GridResult:
    import_kw: Mapping[str | int, pd.Series]
    export_kw: Mapping[str | int, pd.Series]
    q_kvar: Mapping[str | int, pd.Series]
    cost: pd.Series

    def __post_init__(self):
        phases = normalize_phases(self.import_kw)
        self.import_kw, index = _aligned_profiles(
            self.import_kw, phases, label="grid import result"
        )
        self.export_kw, _ = _aligned_profiles(
            self.export_kw, phases, label="grid export result", index=index
        )
        self.q_kvar, _ = _aligned_profiles(
            self.q_kvar, phases, label="grid reactive result", index=index
        )
        self.cost = self.cost.astype(float).copy()
        if not self.cost.index.equals(index):
            raise ValueError("Grid cost does not match the result horizon")
        if self.cost.isna().any() or not self.cost.map(math.isfinite).all():
            raise ValueError("Grid cost must contain finite values")
        if any((series < 0.0).any() for series in (*self.import_kw.values(), *self.export_kw.values())):
            raise ValueError("Grid import and export cannot be negative")

    @property
    def import_total_kw(self):
        index = next(iter(self.import_kw.values())).index
        return sum(self.import_kw.values(), start=pd.Series(0.0, index=index))

    @property
    def export_total_kw(self):
        index = next(iter(self.export_kw.values())).index
        return sum(self.export_kw.values(), start=pd.Series(0.0, index=index))


@dataclass
class BessResult:
    p_net_kw: Mapping[str | int, pd.Series]
    q_injection_kvar: Mapping[str | int, pd.Series]
    soc_kwh: pd.Series
    soc_frac: pd.Series

    def __post_init__(self):
        phases = normalize_phases(self.p_net_kw)
        self.p_net_kw, index = _aligned_profiles(
            self.p_net_kw, phases, label="BESS active result"
        )
        self.q_injection_kvar, _ = _aligned_profiles(
            self.q_injection_kvar, phases, label="BESS reactive result", index=index
        )
        self.soc_kwh = self.soc_kwh.astype(float).copy()
        self.soc_frac = self.soc_frac.astype(float).copy()
        if not self.soc_kwh.index.equals(index) or not self.soc_frac.index.equals(index):
            raise ValueError("BESS SoC results do not match the phase-result horizon")
        tolerance = 1e-7
        if (self.soc_kwh < -tolerance).any() or (
            (self.soc_frac < -tolerance) | (self.soc_frac > 1.0 + tolerance)
        ).any():
            raise ValueError("Invalid BESS SoC result")
        self.soc_kwh = self.soc_kwh.clip(lower=0.0)
        self.soc_frac = self.soc_frac.clip(lower=0.0, upper=1.0)

    @property
    def p_net_total_kw(self):
        index = next(iter(self.p_net_kw.values())).index
        return sum(self.p_net_kw.values(), start=pd.Series(0.0, index=index))


@dataclass
class PvResult:
    available_kw: Mapping[str | int, pd.Series]
    generation_kw: Mapping[str | int, pd.Series]
    q_injection_kvar: Mapping[str | int, pd.Series]
    curtailment_kw: Mapping[str | int, pd.Series]

    def __post_init__(self):
        phases = normalize_phases(self.available_kw)
        self.available_kw, index = _aligned_profiles(
            self.available_kw, phases, label="PV availability result"
        )
        self.generation_kw, _ = _aligned_profiles(
            self.generation_kw, phases, label="PV generation result", index=index
        )
        self.q_injection_kvar, _ = _aligned_profiles(
            self.q_injection_kvar, phases, label="PV reactive result", index=index
        )
        self.curtailment_kw, _ = _aligned_profiles(
            self.curtailment_kw, phases, label="PV curtailment result", index=index
        )
        nonnegative = (
            *self.available_kw.values(),
            *self.generation_kw.values(),
            *self.curtailment_kw.values(),
        )
        if any((series < 0.0).any() for series in nonnegative):
            raise ValueError("PV availability, generation and curtailment cannot be negative")

    @property
    def generation_total_kw(self):
        index = next(iter(self.generation_kw.values())).index
        return sum(self.generation_kw.values(), start=pd.Series(0.0, index=index))
