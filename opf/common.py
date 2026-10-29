"""Shared components, case configuration, time helpers and PV curve defaults."""
from dataclasses import dataclass, field
import math
from pathlib import Path

import pandas as pd


CASE_SCHEMA_VERSION = 1
OBSERVATION_SCHEMA_VERSION = 1

VV_V = (0.90, 0.92, 0.98, 1.02, 1.08, 1.15)
VV_Q = (1.0, 1.0, 0.0, 0.0, -1.0, -1.0)
VW_V = (0.90, 1.06, 1.10, 1.15)
VW_P = (1.0, 1.0, 0.0, 0.0)
QMAX_FRAC = 0.44
V_MAX_EMERG = 1.10


@dataclass(eq=False)
class Base:
    """System bases and per-unit conversions."""

    s_base_kva: float
    v_base_kv: float
    z_base_ohm: float = field(init=False)

    def __post_init__(self):
        if not math.isfinite(self.s_base_kva) or self.s_base_kva <= 0.0:
            raise ValueError("s_base_kva must be finite and positive")
        if not math.isfinite(self.v_base_kv) or self.v_base_kv <= 0.0:
            raise ValueError("v_base_kv must be finite and positive")
        self.z_base_ohm = self.v_base_kv ** 2 * 1e3 / self.s_base_kva

    def pu_power(self, kw): return kw / self.s_base_kva
    def pu_energy(self, kwh): return kwh / self.s_base_kva
    def pu_impedance(self, ohm): return ohm / self.z_base_ohm
    def to_kw(self, pu): return pu * self.s_base_kva
    def to_kwh(self, pu): return pu * self.s_base_kva


@dataclass(eq=False)
class Bess:
    id: str
    bus: int
    e_cap_kwh: float
    p_charge_max_kw: float
    p_discharge_max_kw: float
    eta_charge: float
    eta_discharge: float
    soc_init_frac: float
    soc_min_frac: float
    soc_max_frac: float
    cyclic_soc: bool
    s_max_kva: float | None = None
    reactive_control: bool = False
    q_loss_rated_kw: float = 0.0
    soc_terminal_frac: float | None = None
    state: object = field(default=None, init=False)
    action: object = field(default=None, init=False)
    result: object = field(default=None, init=False)

    def __post_init__(self):
        if self.s_max_kva is None:
            self.s_max_kva = max(self.p_charge_max_kw, self.p_discharge_max_kw)


@dataclass(eq=False)
class Pv:
    id: str
    bus: int
    p_max_kw: float
    s_max_kva: float
    control: str
    curtailable: bool
    power_factor: float
    avail_kw: pd.Series
    q_loss_rated_kw: float = 0.0
    night_var: bool = False
    state: object = field(default=None, init=False)
    action: object = field(default=None, init=False)
    result: object = field(default=None, init=False)


@dataclass(eq=False)
class Summary:
    status: str
    objective_cost: float
    energy_cost: float
    energy_import_kwh: float
    energy_export_kwh: float
    pv_generated_kwh: float
    pv_curtailed_kwh: float
    losses_kwh: float
    v_min_pu: float
    v_max_pu: float
    socp_gap_max_pu2: float
    socp_gap_max_normalized: float
    socp_gap_tolerance: float
    socp_tightness_margin: float
    socp_tightness: str
    socp_relaxation_tight: bool
    inverter_losses_kwh: float = 0.0
    socp_gap_max_relative_flow: float = 0.0
    socp_current_error_max_a: float = 0.0
    socp_loss_error_max_w: float = 0.0

    @property
    def socp_confidence_margin(self):
        return self.socp_tightness_margin

    @property
    def socp_confidence(self):
        return self.socp_tightness

    def as_dict(self): return dict(self.__dict__)


def infer_dt_hours(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 1.0
    differences = idx.to_series().diff().dropna().dt.total_seconds() / 3600.0
    if (differences <= 0.0).any() or not all(
        math.isclose(value, differences.iloc[0], rel_tol=0.0, abs_tol=1e-12)
        for value in differences
    ):
        raise ValueError("demand timestamps must be increasing and equally spaced")
    return float(differences.iloc[0])


def resolve_case_source(source: str | Path) -> tuple[Path, Path]:
    resolved = Path(source).expanduser().resolve()
    if resolved.is_dir():
        root = resolved
        config_path = root / "config.json"
    elif resolved.is_file():
        root = resolved.parent
        config_path = resolved
    else:
        raise FileNotFoundError(f"Case source not found: {resolved}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Case configuration not found: {config_path}")
    return root, config_path


def validate_case_config(config: dict) -> None:
    version = config.get("schema_version", CASE_SCHEMA_VERSION)
    if isinstance(version, bool) or version != CASE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported case schema_version {version!r}; "
            f"expected {CASE_SCHEMA_VERSION}"
        )
