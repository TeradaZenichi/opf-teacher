"""Pre-action observations and commands for phase-native teachers."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

from opf.three_phase.phases import normalize_phases, phase_values


@dataclass
class BusState:
    v_before_pu: Mapping[str | int, float]
    angle_before_deg: Mapping[str | int, float]
    p_load_kw: Mapping[str | int, float]
    q_load_kvar: Mapping[str | int, float]

    def __post_init__(self):
        phases = normalize_phases(self.v_before_pu)
        self.v_before_pu = phase_values(self.v_before_pu, phases, label="bus voltage")
        self.angle_before_deg = phase_values(
            self.angle_before_deg, phases, label="bus voltage angle"
        )
        self.p_load_kw = phase_values(self.p_load_kw, phases, label="bus active load")
        self.q_load_kvar = phase_values(
            self.q_load_kvar, phases, label="bus reactive load"
        )
        if any(value <= 0.0 for value in self.v_before_pu.values()):
            raise ValueError("Observed phase voltages must be positive")

    @property
    def phases(self):
        return tuple(self.v_before_pu)


@dataclass
class BessState:
    soc_before_frac: float
    previous_p_kw: Mapping[str | int, float]
    previous_q_kvar: Mapping[str | int, float]

    def __post_init__(self):
        phases = normalize_phases(self.previous_p_kw)
        self.previous_p_kw = phase_values(
            self.previous_p_kw, phases, label="previous BESS active power"
        )
        self.previous_q_kvar = phase_values(
            self.previous_q_kvar, phases, label="previous BESS reactive power"
        )
        self.soc_before_frac = float(self.soc_before_frac)
        if not 0.0 <= self.soc_before_frac <= 1.0:
            raise ValueError("Observed BESS SoC must be between zero and one")

    @property
    def phases(self):
        return tuple(self.previous_p_kw)


@dataclass
class PvState:
    available_kw: Mapping[str | int, float]
    previous_p_kw: Mapping[str | int, float]
    previous_q_kvar: Mapping[str | int, float]

    def __post_init__(self):
        phases = normalize_phases(self.available_kw)
        self.available_kw = phase_values(self.available_kw, phases, label="PV availability")
        self.previous_p_kw = phase_values(
            self.previous_p_kw, phases, label="previous PV active power"
        )
        self.previous_q_kvar = phase_values(
            self.previous_q_kvar, phases, label="previous PV reactive power"
        )
        if any(value < 0.0 for value in self.available_kw.values()):
            raise ValueError("PV availability cannot be negative")

    @property
    def phases(self):
        return tuple(self.available_kw)


@dataclass
class GridState:
    buy_price_per_kwh: float
    sell_price_per_kwh: float

    def __post_init__(self):
        self.buy_price_per_kwh = float(self.buy_price_per_kwh)
        self.sell_price_per_kwh = float(self.sell_price_per_kwh)


@dataclass
class BessAction:
    p_net_kw: Mapping[str | int, float]
    q_injection_kvar: Mapping[str | int, float]

    def __post_init__(self):
        phases = normalize_phases(self.p_net_kw)
        self.p_net_kw = phase_values(self.p_net_kw, phases, label="BESS active command")
        self.q_injection_kvar = phase_values(
            self.q_injection_kvar, phases, label="BESS reactive command"
        )

    @property
    def p_net_total_kw(self):
        return sum(self.p_net_kw.values())

    @property
    def q_injection_total_kvar(self):
        return sum(self.q_injection_kvar.values())


@dataclass
class PvAction:
    generation_kw: Mapping[str | int, float]
    q_injection_kvar: Mapping[str | int, float]

    def __post_init__(self):
        phases = normalize_phases(self.generation_kw)
        self.generation_kw = phase_values(self.generation_kw, phases, label="PV generation")
        self.q_injection_kvar = phase_values(
            self.q_injection_kvar, phases, label="PV reactive command"
        )
        if any(value < 0.0 for value in self.generation_kw.values()):
            raise ValueError("PV generation command cannot be negative")

    @property
    def generation_total_kw(self):
        return sum(self.generation_kw.values())

    @property
    def q_injection_total_kvar(self):
        return sum(self.q_injection_kvar.values())


def state_values(state):
    if state is None:
        raise ValueError("Missing pre-action observation; call Teacher.observe first")
    return deepcopy(vars(state))


def local_state_action(device, bus):
    if bus.id != device.connection.bus:
        raise ValueError(f"Device {device.id!r} is not connected to bus {bus.id!r}")
    if device.action is None:
        raise ValueError("No teacher action; solve the observed case first")
    x = {"bus": state_values(bus.state), "device": state_values(device.state)}
    return x, state_values(device.action)
