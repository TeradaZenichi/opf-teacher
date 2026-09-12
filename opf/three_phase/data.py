"""Load phase-indexed cases for the unbalanced three-phase formulation."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import pandas as pd

from opf.components import Base, Bess, Pv
from opf.case_source import resolve_case_source, validate_case_config
from opf.data import _infer_dt_hours
from opf.opendss import load_opendss_network, resolve_bus_id
from opf.three_phase.components import (
    Branch,
    Bus,
    Case,
    ConnectedBess,
    ConnectedPv,
    DeviceConnection,
    Grid,
)
from opf.three_phase.phases import normalize_phases


_DEMAND_COLUMN = re.compile(r"^(P|Q)(.+)_([abcABC123])$")


def load_case(path: str | Path) -> Case:
    path, config_path = resolve_case_source(path)
    cfg = _read_json(config_path)
    validate_case_config(cfg)
    files = cfg.get("files", {})
    demand_path = path / files.get("demand", "demand.csv")
    price_path = path / files.get("prices", "price.csv")
    devices_path = path / files.get("devices", "devices.json")
    configured = str(cfg.get("formulation", "three_phase_ivr")).strip().lower()
    if configured not in {"three_phase_ivr", "unbalanced_ac_ivr"}:
        raise ValueError("A three-phase case must declare formulation='three_phase_ivr'")

    base = Base(float(cfg["base"]["s_base_kva"]), float(cfg["base"]["v_base_kv"]))
    network_cfg = cfg.get("network", {})
    master_ref = network_cfg.get("master")
    if not master_ref:
        raise ValueError("OpenDSS network requires network.master in config.json")
    network = load_opendss_network(
        path / master_ref,
        slack_bus=network_cfg.get("slack_bus", cfg.get("grid", {}).get("bus")),
        bus_ids=network_cfg.get("bus_ids"),
        fallback_v_base_kv=base.v_base_kv,
    )
    bus_name_to_id = network.bus_name_to_id

    def case_bus(value) -> int:
        return resolve_bus_id(value, bus_name_to_id)

    demand = pd.read_csv(demand_path, parse_dates=["timestamp"])
    demand = demand.sort_values("timestamp").reset_index(drop=True)
    index = pd.DatetimeIndex(demand["timestamp"])
    phase_loads = _read_phase_demand(demand, index, case_bus)

    limits = cfg.get("voltage_limits", {})
    v_min = limits.get("v_min_pu", 0.95)
    v_max = limits.get("v_max_pu", 1.05)
    buses = {}
    for source in network.buses:
        phases = normalize_phases(source.phases)
        p_by_phase = {}
        q_by_phase = {}
        for phase in phases:
            key = (source.id, phase)
            if key not in phase_loads["P"] or key not in phase_loads["Q"]:
                raise ValueError(
                    f"demand.csv must explicitly define P and Q for bus "
                    f"{source.name!r}, phase {phase!r}"
                )
            p_by_phase[phase] = phase_loads["P"][key]
            q_by_phase[phase] = phase_loads["Q"][key]
        buses[source.id] = Bus(
            id=source.id,
            name=source.name,
            phases=phases,
            p_load_kw=p_by_phase,
            q_load_kvar=q_by_phase,
            v_min_pu=v_min,
            v_max_pu=v_max,
            kv_base_ln=source.kv_base_ln,
        )

    branches = []
    for source in network.branches:
        if source.element_type != "line":
            raise ValueError(
                f"Three-phase IVR currently supports lines only; got "
                f"{source.element_type!r} for {source.name!r}"
            )
        phases = normalize_phases(source.phases)
        size = len(phases)
        if len(source.r_matrix_ohm) != size or len(source.x_matrix_ohm) != size:
            raise ValueError(f"Branch {source.name!r} requires complete Rmatrix and Xmatrix")
        impedance = tuple(
            tuple(
                complex(source.r_matrix_ohm[row][column], source.x_matrix_ohm[row][column])
                for column in range(size)
            )
            for row in range(size)
        )
        branches.append(Branch(
            name=source.name,
            from_bus=source.from_bus,
            to_bus=source.to_bus,
            phases=phases,
            z_matrix_ohm=impedance,
            norm_amps=source.norm_amps,
            element_type=source.element_type,
            connections=source.connections,
        ))

    price_frame = _read_timed_csv(price_path)
    _require_index(price_frame, index, str(price_path.name))
    price = pd.Series(price_frame["price_per_kwh"].to_numpy(), index=index, dtype=float)

    grid_cfg = cfg["grid"]
    grid_bus = network.root_bus if "bus" not in grid_cfg else case_bus(grid_cfg["bus"])
    grid_phases = buses[grid_bus].phases
    grid = Grid(
        bus=grid_bus,
        phases=grid_phases,
        v_ref_pu=_phase_setting(grid_cfg.get("v_ref_pu", 1.0), grid_phases),
        angle_ref_deg=_phase_setting(
            grid_cfg.get("angle_ref_deg", {"a": 0.0, "b": -120.0, "c": 120.0}),
            grid_phases,
        ),
        p_import_max_kw=float(grid_cfg.get("p_import_max_kw", 0.0)),
        p_export_max_kw=float(grid_cfg.get("p_export_max_kw", 0.0)),
        q_max_kvar=float(grid_cfg.get("q_max_kvar", 0.0)),
        feed_in_ratio=float(grid_cfg.get("feed_in_tariff_ratio", 1.0)),
    )

    device_cfg = _read_json(devices_path) if devices_path.exists() else {}
    bess = [_load_bess(item, case_bus) for item in device_cfg.get("bess", [])]
    pv = [_load_pv(item, path, index, case_bus) for item in device_cfg.get("pv", [])]

    case = Case(
        name=cfg.get("name", path.name),
        base=base,
        buses=buses,
        branches=branches,
        grid=grid,
        bess=bess,
        pv=pv,
        timestamps=list(index),
        dt_h=_infer_dt_hours(index),
        price=price,
    )
    case.source_root = path
    case.config_path = config_path
    return case


def _read_phase_demand(frame, index, case_bus):
    result = {"P": {}, "Q": {}}
    for column in frame.columns:
        if column == "timestamp":
            continue
        match = _DEMAND_COLUMN.fullmatch(column)
        if match is None:
            raise ValueError(
                f"Invalid three-phase demand column {column!r}; expected "
                "P<bus>_<phase> or Q<bus>_<phase>"
            )
        kind, bus_name, phase = match.groups()
        phase_name = normalize_phases((phase,))[0]
        key = (case_bus(bus_name.lstrip("_")), phase_name)
        if key in result[kind]:
            raise ValueError(f"Duplicate {kind} demand for bus/phase {key}")
        result[kind][key] = pd.Series(frame[column].to_numpy(), index=index, dtype=float)
    return result


def _load_bess(item, case_bus):
    bus = case_bus(item["bus"])
    device = Bess(
        id=str(item["id"]),
        bus=bus,
        e_cap_kwh=float(item["e_cap_kwh"]),
        p_charge_max_kw=float(item["p_charge_max_kw"]),
        p_discharge_max_kw=float(item["p_discharge_max_kw"]),
        eta_charge=float(item["eta_charge"]),
        eta_discharge=float(item["eta_discharge"]),
        soc_init_frac=float(item["soc_init_frac"]),
        soc_min_frac=float(item.get("soc_min_frac", 0.0)),
        soc_max_frac=float(item.get("soc_max_frac", 1.0)),
        cyclic_soc=bool(item.get("cyclic_soc", True)),
        s_max_kva=float(item.get("s_max_kva", max(item["p_charge_max_kw"], item["p_discharge_max_kw"]))),
        reactive_control=bool(item.get("reactive_control", False)),
        q_loss_rated_kw=float(item.get("q_loss_rated_kw", 0.0)),
        soc_terminal_frac=(
            float(item["soc_terminal_frac"])
            if item.get("soc_terminal_frac") is not None else None
        ),
    )
    return ConnectedBess(
        device=device,
        connection=DeviceConnection(
            bus=bus,
            phases=tuple(item["phases"]),
            connection=item.get("connection", "wye"),
            dispatch_mode=item.get("dispatch_mode", "aggregate"),
        ),
    )


def _load_pv(item, path, index, case_bus):
    bus = case_bus(item["bus"])
    total = _load_profile(path, item["profile"], index)
    device = Pv(
        id=str(item["id"]),
        bus=bus,
        p_max_kw=float(item["p_max_kw"]),
        s_max_kva=float(item.get("s_max_kva", item["p_max_kw"])),
        control=str(item.get("control", "optimal")),
        curtailable=bool(item.get("curtailable", True)),
        power_factor=float(item.get("power_factor", 1.0)),
        avail_kw=total,
        q_loss_rated_kw=float(item.get("q_loss_rated_kw", 0.0)),
        night_var=bool(item.get("night_var", False)),
    )
    connection = DeviceConnection(
        bus=bus,
        phases=tuple(item["phases"]),
        connection=item.get("connection", "wye"),
        dispatch_mode=item.get("dispatch_mode", "aggregate"),
    )
    references = item.get("phase_profiles", {})
    expected = set(connection.phases)
    normalized_refs = {normalize_phases((phase,))[0]: ref for phase, ref in references.items()}
    if set(normalized_refs) != expected:
        raise ValueError(f"PV {device.id!r} phase_profiles must define {tuple(connection.phases)}")
    available = {
        phase: _load_profile(path, normalized_refs[phase], index)
        for phase in connection.phases
    }
    return ConnectedPv(device=device, connection=connection, available_kw=available)


def _load_profile(path, reference, index):
    filename, separator, column = str(reference).partition(":")
    if not separator:
        raise ValueError(f"Invalid profile reference {reference!r}; expected file.csv:column")
    frame = _read_timed_csv(path / filename)
    _require_index(frame, index, filename)
    return pd.Series(frame[column].to_numpy(), index=index, dtype=float)


def _read_timed_csv(path):
    return pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _require_index(frame, expected, label):
    actual = pd.DatetimeIndex(frame["timestamp"])
    if not actual.equals(expected):
        raise ValueError(f"{label} timestamps must exactly match demand.csv")


def _phase_setting(value, phases):
    if isinstance(value, dict):
        return value
    value = float(value)
    return {phase: value for phase in phases}


def _read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)
