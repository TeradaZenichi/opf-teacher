"""Single-phase network, result and case objects."""
from dataclasses import dataclass, field
from copy import deepcopy
import math

import pandas as pd

from opf_teacher.opf.common import Base
from opf_teacher.opf.single_phase.states import local_state_action, state_values


def phase_count(phases) -> int:
    """Default to three phases when phase data is absent."""
    count = len(tuple(phases or ())) or 3
    if count not in {1, 3}:
        raise ValueError(
            f"Unsupported phase count {count}; use a single-phase equivalent "
            "or a balanced three-phase representation"
        )
    return count


def phase_power_factor(phases) -> float:
    """Factor relating terminal voltage and current to total apparent power."""
    return 1.0 if phase_count(phases) == 1 else math.sqrt(3.0)


def system_voltage_base_kv(kv_base: float | None, phases, fallback_kv: float) -> float:
    if phase_count(phases) == 1:
        return float(fallback_kv)
    bus_kv = float(kv_base or 0.0)
    if bus_kv <= 0.0:
        return float(fallback_kv)
    # Balanced three-phase models use the line-to-line voltage base.
    return math.sqrt(3.0) * bus_kv


@dataclass(eq=False)
class BusResult:
    v_pu: object


@dataclass(eq=False)
class BranchResult:
    p_kw: object
    q_kvar: object
    loss_kw: object
    socp_gap_pu2: object
    socp_gap_normalized: object
    socp_gap_relative_flow: object = None
    socp_current_error_a: object = None
    socp_loss_error_w: object = None


@dataclass(eq=False)
class GridResult:
    import_kw: object
    export_kw: object
    q_kvar: object
    cost: object


@dataclass(eq=False)
class BessResult:
    charge_kw: object
    discharge_kw: object
    p_net_kw: object
    soc_kwh: object
    soc_frac: object
    q_kvar: object = None
    inverter_loss_kw: object = None


@dataclass(eq=False)
class PvResult:
    avail_kw: object
    gen_kw: object
    curtail_kw: object
    q_kvar: object
    inverter_loss_kw: object = None
    p_net_kw: object = None
    grid_consumption_kw: object = None


@dataclass(eq=False)
class Bus:
    id: int
    type: str
    name: str
    v_min_pu: float
    v_max_pu: float
    p_load_kw: pd.Series
    q_load_kw: pd.Series
    phases: tuple = ()
    kv_base_ln: float | None = None
    state: object = field(default=None, init=False)
    result: object = field(default=None, init=False)

    def __post_init__(self):
        self.phases = tuple(self.phases or ())

    @property
    def is_slack(self): return self.type == "slack"


@dataclass(eq=False)
class Branch:
    from_bus: int
    to_bus: int
    r_ohm: float
    x_ohm: float
    s_max_kva: float
    name: str | None = None
    phases: tuple = ()
    norm_amps: float | None = None
    r_matrix_ohm: object = None
    x_matrix_ohm: object = None
    length: float | None = None
    length_units: str | None = None
    element_type: str = "line"
    tap_ratio: float = 1.0
    r_pu_on_rating: float | None = None
    x_pu_on_rating: float | None = None
    impedance_base_kva: float | None = None
    connections: tuple = ()
    result: object = field(default=None, init=False)

    def __post_init__(self):
        self.phases = tuple(self.phases or ())
        self.connections = tuple(self.connections or ())

    def impedance_pu(self, base, from_bus):
        if self.impedance_base_kva is not None:
            scale = base.s_base_kva / self.impedance_base_kva
            return self.r_pu_on_rating * scale, self.x_pu_on_rating * scale

        voltage_base_kv = system_voltage_base_kv(from_bus.kv_base_ln, self.phases, base.v_base_kv)
        z_base_ohm = voltage_base_kv ** 2 * 1e3 / base.s_base_kva
        return self.r_ohm / z_base_ohm, self.x_ohm / z_base_ohm


@dataclass(eq=False)
class Grid:
    bus: int
    v_ref_pu: float
    p_import_max_kw: float
    p_export_max_kw: float
    q_max_kvar: float
    feed_in_ratio: float
    state: object = field(default=None, init=False)
    result: object = field(default=None, init=False)


@dataclass(eq=False)
class Case:
    name: str
    base: Base
    buses: dict
    branches: list
    grid: Grid
    bess: list
    pv: list
    timestamps: list
    dt_h: float
    price: pd.Series
    parent_branch: dict = field(default_factory=dict, init=False)
    children: dict = field(default_factory=dict, init=False)
    bus_name_to_id: dict = field(default_factory=dict, init=False)
    summary: object = field(default=None, init=False)
    formulation: str | None = field(default=None, init=False)
    quality: dict | None = field(default=None, init=False)
    source_root: object = field(default=None, init=False)
    config_path: object = field(default=None, init=False)

    @property
    def periods(self): return range(len(self.timestamps))
    @property
    def n_periods(self): return len(self.timestamps)
    @property
    def root(self): return self.grid.bus
    @property
    def index(self): return pd.DatetimeIndex(self.timestamps)

    def horizon(self, start: int, periods: int | None = None):
        """Copy a forecast horizon without observations, actions or results."""
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < self.n_periods:
            raise ValueError("start must identify a period in the case")
        stop = self.n_periods
        if periods is not None:
            if isinstance(periods, bool) or not isinstance(periods, int) or periods < 1:
                raise ValueError("periods must be a positive integer")
            stop = start + periods
            if stop > self.n_periods:
                raise ValueError("The requested horizon exceeds the case")

        case = deepcopy(self)
        case.timestamps = case.timestamps[start:stop]
        case.price = case.price.iloc[start:stop].copy()
        for bus in case.buses.values():
            bus.p_load_kw = bus.p_load_kw.iloc[start:stop].copy()
            bus.q_load_kw = bus.q_load_kw.iloc[start:stop].copy()
            bus.state = None
            bus.result = None
        for branch in case.branches:
            branch.result = None
        for devices in (case.bess, case.pv):
            for device in devices:
                device.state = None
                device.action = None
                device.result = None
        for device in case.pv:
            device.avail_kw = device.avail_kw.iloc[start:stop].copy()
        case.grid.state = None
        case.grid.result = None
        case.summary = None
        case.formulation = None
        case.quality = None
        return case

    def state_action(self):
        """Return central X/Y for the first interval, keyed by bus and device ID."""
        if self.summary is None:
            raise ValueError("No teacher solution; solve the observed case first")
        x = {
            "timestamp": self.index[0].isoformat(),
            "dt_h": self.dt_h,
            "buses": {bid: state_values(bus.state) for bid, bus in sorted(self.buses.items())},
            "grid": state_values(self.grid.state),
            "bess": {},
            "pv": {},
        }
        y = {"bess": {}, "pv": {}}
        for kind, devices in (("bess", self.bess), ("pv", self.pv)):
            for device in sorted(devices, key=lambda d: d.id):
                local_x, action = local_state_action(device, self.buses[device.bus])
                x[kind][device.id] = local_x["device"]
                y[kind][device.id] = action
        return x, y
