"""Phase-native network and device contracts for unbalanced OPF."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

import pandas as pd

from opf.components import Base, Bess, Pv
from opf.three_phase.phases import normalize_phase, normalize_phases, phase_values
from opf.three_phase.states import local_state_action, state_values


def _profile_map(
    values: Mapping[str | int, pd.Series],
    phases: Sequence[str],
    *,
    label: str,
) -> dict[str, pd.Series]:
    normalized = {normalize_phase(key): value for key, value in values.items()}
    if set(normalized) != set(phases):
        raise ValueError(
            f"{label} phases must be exactly {tuple(phases)}; got {tuple(normalized)}"
        )

    result = {}
    reference_index = None
    for phase in phases:
        series = normalized[phase]
        if not isinstance(series, pd.Series):
            raise TypeError(f"{label}[{phase!r}] must be a pandas Series")
        numeric = series.astype(float)
        if numeric.isna().any() or not numeric.map(math.isfinite).all():
            raise ValueError(f"{label}[{phase!r}] must contain finite values")
        if reference_index is None:
            reference_index = numeric.index
        elif not numeric.index.equals(reference_index):
            raise ValueError(f"All {label} phase profiles must use the same index")
        result[phase] = numeric.copy()
    return result


def _square_complex_matrix(values, size: int, *, label: str):
    rows = tuple(tuple(complex(value) for value in row) for row in values)
    if len(rows) != size or any(len(row) != size for row in rows):
        raise ValueError(f"{label} must be a {size}x{size} matrix")
    if not all(
        math.isfinite(value.real) and math.isfinite(value.imag)
        for row in rows for value in row
    ):
        raise ValueError(f"{label} must contain finite values")
    return rows


@dataclass
class Bus:
    id: int
    name: str
    phases: tuple[str, ...]
    p_load_kw: Mapping[str | int, pd.Series]
    q_load_kvar: Mapping[str | int, pd.Series]
    v_min_pu: float | Mapping[str | int, float] = 0.95
    v_max_pu: float | Mapping[str | int, float] = 1.05
    kv_base_ln: float | None = None
    state: object | None = field(default=None, init=False)
    result: object | None = field(default=None, init=False)

    def __post_init__(self):
        self.phases = normalize_phases(self.phases)
        self.p_load_kw = _profile_map(
            self.p_load_kw, self.phases, label=f"bus {self.id} active load"
        )
        self.q_load_kvar = _profile_map(
            self.q_load_kvar, self.phases, label=f"bus {self.id} reactive load"
        )
        if not self.index.equals(next(iter(self.q_load_kvar.values())).index):
            raise ValueError(f"Bus {self.id} active and reactive profiles must align")

        self.v_min_pu = self._voltage_limits(self.v_min_pu, "minimum voltage")
        self.v_max_pu = self._voltage_limits(self.v_max_pu, "maximum voltage")
        for phase in self.phases:
            if not 0.0 < self.v_min_pu[phase] < self.v_max_pu[phase]:
                raise ValueError(f"Invalid voltage bounds at bus {self.id}, phase {phase}")
        if self.kv_base_ln is not None and (
            not math.isfinite(float(self.kv_base_ln)) or float(self.kv_base_ln) <= 0.0
        ):
            raise ValueError("kv_base_ln must be positive when provided")

    def _voltage_limits(self, values, label):
        if isinstance(values, Mapping):
            return phase_values(values, self.phases, label=f"bus {self.id} {label}")
        value = float(values)
        if not math.isfinite(value):
            raise ValueError(f"Bus {self.id} {label} must be finite")
        return {phase: value for phase in self.phases}

    @property
    def index(self) -> pd.Index:
        return next(iter(self.p_load_kw.values())).index

    @property
    def p_load_total_kw(self) -> pd.Series:
        return sum(self.p_load_kw.values(), start=pd.Series(0.0, index=self.index))

    @property
    def q_load_total_kvar(self) -> pd.Series:
        return sum(self.q_load_kvar.values(), start=pd.Series(0.0, index=self.index))


@dataclass
class Branch:
    name: str
    from_bus: int
    to_bus: int
    phases: tuple[str, ...]
    z_matrix_ohm: Sequence[Sequence[complex]]
    norm_amps: float | Mapping[str | int, float]
    tap_matrix: Sequence[Sequence[complex]] | None = None
    element_type: str = "line"
    connections: tuple[str, ...] = ()
    result: object | None = field(default=None, init=False)

    def __post_init__(self):
        self.phases = normalize_phases(self.phases)
        size = len(self.phases)
        self.z_matrix_ohm = _square_complex_matrix(
            self.z_matrix_ohm, size, label=f"branch {self.name} impedance"
        )
        if self.tap_matrix is None:
            self.tap_matrix = tuple(
                tuple(1.0 + 0.0j if row == col else 0.0j for col in range(size))
                for row in range(size)
            )
        else:
            self.tap_matrix = _square_complex_matrix(
                self.tap_matrix, size, label=f"branch {self.name} tap"
            )

        if isinstance(self.norm_amps, Mapping):
            ratings = phase_values(
                self.norm_amps, self.phases, label=f"branch {self.name} ampacity"
            )
        else:
            rating = float(self.norm_amps)
            ratings = {phase: rating for phase in self.phases}
        if any(value <= 0.0 for value in ratings.values()):
            raise ValueError(f"Branch {self.name} ampacity must be positive")
        self.norm_amps = ratings

        if self.connections:
            if len(self.connections) != 2:
                raise ValueError("Branch connections must describe both terminals")
            self.connections = tuple(str(value).strip().lower() for value in self.connections)
            if any(value not in {"wye", "delta"} for value in self.connections):
                raise ValueError("Branch connections must be 'wye' or 'delta'")


@dataclass(frozen=True)
class DeviceConnection:
    bus: int
    phases: tuple[str, ...]
    connection: str = "wye"
    dispatch_mode: str = "aggregate"

    def __post_init__(self):
        phases = normalize_phases(self.phases)
        connection = str(self.connection).strip().lower()
        dispatch = str(self.dispatch_mode).strip().lower()
        if connection not in {"wye", "delta"}:
            raise ValueError("Device connection must be 'wye' or 'delta'")
        if connection == "delta" and len(phases) < 2:
            raise ValueError("A delta device requires at least two phases")
        if dispatch not in {"aggregate", "per_phase"}:
            raise ValueError("dispatch_mode must be 'aggregate' or 'per_phase'")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "connection", connection)
        object.__setattr__(self, "dispatch_mode", dispatch)


@dataclass
class ConnectedBess:
    device: Bess
    connection: DeviceConnection
    state: object | None = field(default=None, init=False)
    action: object | None = field(default=None, init=False)
    result: object | None = field(default=None, init=False)

    def __post_init__(self):
        if self.device.bus != self.connection.bus:
            raise ValueError(f"BESS {self.device.id!r} connection uses a different bus")

    @property
    def id(self):
        return self.device.id

    def state_action(self, bus):
        return local_state_action(self, bus)


@dataclass
class ConnectedPv:
    device: Pv
    connection: DeviceConnection
    available_kw: Mapping[str | int, pd.Series]
    state: object | None = field(default=None, init=False)
    action: object | None = field(default=None, init=False)
    result: object | None = field(default=None, init=False)

    def __post_init__(self):
        if self.device.bus != self.connection.bus:
            raise ValueError(f"PV {self.device.id!r} connection uses a different bus")
        self.available_kw = _profile_map(
            self.available_kw,
            self.connection.phases,
            label=f"PV {self.device.id!r} availability",
        )
        aggregate = self.available_total_kw
        legacy = self.device.avail_kw.astype(float)
        if not aggregate.index.equals(legacy.index):
            raise ValueError(f"PV {self.device.id!r} availability horizons do not align")
        if (aggregate - legacy).abs().max() > 1e-9:
            raise ValueError(
                f"PV {self.device.id!r} phase availability must sum to its total profile"
            )

    @property
    def id(self):
        return self.device.id

    @property
    def available_total_kw(self):
        index = next(iter(self.available_kw.values())).index
        return sum(self.available_kw.values(), start=pd.Series(0.0, index=index))

    def state_action(self, bus):
        return local_state_action(self, bus)


@dataclass
class Grid:
    bus: int
    phases: tuple[str, ...]
    v_ref_pu: Mapping[str | int, float]
    angle_ref_deg: Mapping[str | int, float]
    p_import_max_kw: float
    p_export_max_kw: float
    q_max_kvar: float
    feed_in_ratio: float
    state: object | None = field(default=None, init=False)
    result: object | None = field(default=None, init=False)

    def __post_init__(self):
        self.phases = normalize_phases(self.phases)
        self.v_ref_pu = phase_values(self.v_ref_pu, self.phases, label="grid voltage")
        self.angle_ref_deg = phase_values(
            self.angle_ref_deg, self.phases, label="grid voltage angle"
        )
        limits = (
            self.p_import_max_kw,
            self.p_export_max_kw,
            self.q_max_kvar,
            self.feed_in_ratio,
        )
        if not all(math.isfinite(float(value)) for value in limits):
            raise ValueError("Grid limits and tariff ratio must be finite")
        if min(self.p_import_max_kw, self.p_export_max_kw, self.q_max_kvar) < 0.0:
            raise ValueError("Grid power limits cannot be negative")


@dataclass
class Case:
    name: str
    base: Base
    buses: Mapping[int, Bus]
    branches: Sequence[Branch]
    grid: Grid
    bess: Sequence[ConnectedBess]
    pv: Sequence[ConnectedPv]
    timestamps: Sequence[pd.Timestamp]
    dt_h: float
    price: pd.Series
    summary: object | None = field(default=None, init=False)
    bus_name_to_id: Mapping[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.buses = dict(self.buses)
        self.branches = list(self.branches)
        self.bess = list(self.bess)
        self.pv = list(self.pv)
        self.timestamps = list(pd.DatetimeIndex(self.timestamps))
        self.price = self.price.astype(float).copy()
        self.bus_name_to_id = {bus.name: bus_id for bus_id, bus in self.buses.items()}
        if not self.timestamps:
            raise ValueError("Three-phase case requires at least one timestamp")
        if not math.isfinite(float(self.dt_h)) or self.dt_h <= 0.0:
            raise ValueError("dt_h must be positive")
        if len(self.price) != len(self.timestamps):
            raise ValueError("Price and timestamp horizons must have the same length")
        if not self.price.index.equals(self.index):
            raise ValueError("Price index must match the case timestamps")
        if self.price.isna().any() or not self.price.map(math.isfinite).all():
            raise ValueError("Price values must be finite")

        expected_index = self.index
        for bus_id, bus in self.buses.items():
            if bus_id != bus.id:
                raise ValueError(f"Bus key {bus_id!r} does not match bus id {bus.id!r}")
            if not bus.index.equals(expected_index):
                raise ValueError(f"Bus {bus_id} profiles do not match the case horizon")
        if self.grid.bus not in self.buses:
            raise ValueError("Grid references an unknown bus")
        if not set(self.grid.phases).issubset(self.buses[self.grid.bus].phases):
            raise ValueError("Grid phases are not present at its bus")

        for branch in self.branches:
            if branch.from_bus not in self.buses or branch.to_bus not in self.buses:
                raise ValueError(f"Branch {branch.name!r} references an unknown bus")
            shared = set(self.buses[branch.from_bus].phases) & set(self.buses[branch.to_bus].phases)
            if not set(branch.phases).issubset(shared):
                raise ValueError(f"Branch {branch.name!r} uses phases absent from its buses")

        for devices, label in ((self.bess, "BESS"), (self.pv, "PV")):
            ids = [device.id for device in devices]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{label} IDs must be unique")
            for device in devices:
                connection = device.connection
                if connection.bus not in self.buses:
                    raise ValueError(f"{label} {device.id!r} references an unknown bus")
                if not set(connection.phases).issubset(self.buses[connection.bus].phases):
                    raise ValueError(f"{label} {device.id!r} uses phases absent from its bus")
                if isinstance(device, ConnectedPv) and not device.available_total_kw.index.equals(expected_index):
                    raise ValueError(f"PV {device.id!r} profile does not match the case horizon")

    @property
    def periods(self):
        return range(len(self.timestamps))

    @property
    def n_periods(self):
        return len(self.timestamps)

    @property
    def index(self):
        return pd.DatetimeIndex(self.timestamps)

    @property
    def root(self):
        return self.grid.bus

    def state_action(self):
        """Return phase-native central X/Y for the first interval."""

        if self.summary is None:
            raise ValueError("No teacher solution; solve the observed case first")
        x = {
            "timestamp": self.index[0].isoformat(),
            "dt_h": self.dt_h,
            "phase_order": list(self.grid.phases),
            "buses": {
                bus_id: state_values(bus.state)
                for bus_id, bus in sorted(self.buses.items())
            },
            "grid": state_values(self.grid.state),
            "bess": {},
            "pv": {},
        }
        y = {"bess": {}, "pv": {}}
        for kind in ("bess", "pv"):
            for device in sorted(getattr(self, kind), key=lambda item: item.id):
                local_x, action = device.state_action(
                    self.buses[device.connection.bus]
                )
                x[kind][device.id] = local_x["device"]
                y[kind][device.id] = action
        return x, y
