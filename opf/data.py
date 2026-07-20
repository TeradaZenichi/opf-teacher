"""Leitura dos arquivos de entrada."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from opf.components import Base, Bess, Branch, Bus, Case, Grid, Pv
from opf.opendss import load_opendss_network, resolve_bus_id


def load_case(path: str | Path) -> Case:
    path = Path(path).expanduser().resolve()
    cfg = _read_json(path / "config.json")

    base = Base(float(cfg["base"]["s_base_kva"]), float(cfg["base"]["v_base_kv"]))
    vlim = cfg.get("voltage_limits", {})
    v_min = float(vlim.get("v_min_pu", 0.95))
    v_max = float(vlim.get("v_max_pu", 1.05))
    pf_default = float(cfg.get("defaults", {}).get("power_factor", 1.0))

    network_cfg = cfg.get("network", {})
    master_ref = network_cfg.get("master")
    if not master_ref:
        raise ValueError("OpenDSS network requires network.master in config.json")
    dss_network = load_opendss_network(
        path / master_ref,
        slack_bus=network_cfg.get("slack_bus", cfg.get("grid", {}).get("bus")),
        bus_ids=network_cfg.get("bus_ids"),
        fallback_v_base_kv=base.v_base_kv,
    )
    bus_name_to_id = dss_network.bus_name_to_id

    def case_bus(value) -> int:
        return resolve_bus_id(value, bus_name_to_id)

    dem = (pd.read_csv(path / "demand.csv", parse_dates=["timestamp"])
           .sort_values("timestamp").reset_index(drop=True))
    idx = pd.DatetimeIndex(dem["timestamp"])
    p_load, q_load = {}, {}
    for col in dem.columns:
        if col == "timestamp":
            continue
        kind = col[0].upper()
        if kind not in {"P", "Q"}:
            raise ValueError(f"Invalid demand column {col!r}; expected P<bus> or Q<bus>")
        bus = case_bus(col[1:].lstrip("_"))
        (p_load if kind == "P" else q_load)[bus] = pd.Series(dem[col].to_numpy(), index=idx)

    zero = pd.Series(0.0, index=idx)
    q_ratio = math.tan(math.acos(pf_default))

    buses: dict[int, Bus] = {}
    for dss_bus in dss_network.buses:
        bid = dss_bus.id
        p = p_load.get(bid, zero.copy())
        q = q_load.get(bid, p * q_ratio if bid in p_load else zero.copy())
        buses[bid] = Bus(
            id=bid,
            type="slack" if bid == dss_network.root_bus else "pq",
            name=dss_bus.name,
            v_min_pu=v_min,
            v_max_pu=v_max,
            p_load_kw=p,
            q_load_kw=q,
            phases=dss_bus.phases,
            kv_base_ln=dss_bus.kv_base_ln,
        )
    branches = [
        Branch(
            d.from_bus, d.to_bus, d.r_ohm, d.x_ohm, d.s_max_kva,
            name=d.name,
            phases=d.phases,
            norm_amps=d.norm_amps,
            r_matrix_ohm=d.r_matrix_ohm,
            x_matrix_ohm=d.x_matrix_ohm,
            length=d.length,
            length_units=d.units,
            element_type=d.element_type,
            tap_ratio=d.tap_ratio,
            r_pu_on_rating=d.r_pu_on_rating,
            x_pu_on_rating=d.x_pu_on_rating,
            impedance_base_kva=d.impedance_base_kva,
            connections=d.connections,
        )
        for d in dss_network.branches
    ]

    unknown_profile_buses = (set(p_load) | set(q_load)) - set(buses)
    if unknown_profile_buses:
        raise ValueError(
            f"Demand profiles reference buses not present in the network: "
            f"{sorted(unknown_profile_buses)}"
        )

    g = cfg["grid"]
    grid = Grid(
        bus=dss_network.root_bus if "bus" not in g else case_bus(g["bus"]),
        v_ref_pu=float(g.get("v_ref_pu", 1.0)),
        p_import_max_kw=float(g.get("p_import_max_kw", 0.0)),
        p_export_max_kw=float(g.get("p_export_max_kw", 0.0)),
        q_max_kvar=float(g.get("q_max_kvar", 0.0)),
        feed_in_ratio=float(g.get("feed_in_tariff_ratio", 1.0)),
    )

    price_df = (pd.read_csv(path / "price.csv", parse_dates=["timestamp"])
                .sort_values("timestamp"))
    price = pd.Series(price_df["price_per_kwh"].to_numpy(), index=idx)

    dev = _read_json(path / "devices.json") if (path / "devices.json").exists() else {}

    bess = [
        Bess(
            id=str(d["id"]), bus=case_bus(d["bus"]),
            e_cap_kwh=float(d["e_cap_kwh"]),
            p_charge_max_kw=float(d["p_charge_max_kw"]),
            p_discharge_max_kw=float(d["p_discharge_max_kw"]),
            eta_charge=float(d["eta_charge"]),
            eta_discharge=float(d["eta_discharge"]),
            soc_init_frac=float(d["soc_init_frac"]),
            soc_min_frac=float(d.get("soc_min_frac", 0.0)),
            soc_max_frac=float(d.get("soc_max_frac", 1.0)),
            cyclic_soc=bool(d.get("cyclic_soc", True)),
        )
        for d in dev.get("bess", [])
    ]

    pv = [
        Pv(
            id=str(d["id"]), bus=case_bus(d["bus"]),
            p_max_kw=float(d["p_max_kw"]),
            s_max_kva=float(d.get("s_max_kva", d["p_max_kw"])),
            control=str(d.get("control", "optimal")),
            curtailable=bool(d.get("curtailable", True)),
            power_factor=float(d.get("power_factor", 1.0)),
            avail_kw=_load_profile(path, d["profile"], idx),
        )
        for d in dev.get("pv", [])
    ]

    case = Case(
        name=cfg.get("name", path.name),
        base=base,
        buses=buses,
        branches=branches,
        grid=grid,
        bess=bess,
        pv=pv,
        timestamps=list(idx),
        dt_h=_infer_dt_hours(idx),
        price=price,
    )
    case.bus_name_to_id = dict(bus_name_to_id)
    _build_topology(case)
    _validate(case)
    return case


def _read_json(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _infer_dt_hours(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 1.0
    return float(idx.to_series().diff().dropna().median().total_seconds()) / 3600.0


def _load_profile(path: Path, ref: str, idx: pd.DatetimeIndex) -> pd.Series:
    """Lê uma referência no formato arquivo.csv:coluna."""
    fname, _, col = ref.partition(":")
    df = (pd.read_csv(path / fname, parse_dates=["timestamp"])
          .sort_values("timestamp"))
    return pd.Series(df[col].to_numpy(), index=idx)


def _build_topology(case: Case) -> None:
    case.children = {b: [] for b in case.buses}
    for i, br in enumerate(case.branches):
        case.parent_branch[br.to_bus] = i
        case.children.setdefault(br.from_bus, []).append(br.to_bus)


def _validate(case: Case) -> None:
    if case.n_periods == 0:
        raise ValueError("No periods found in demand.csv")
    if case.root not in case.buses:
        raise ValueError(f"Grid bus {case.root} is not present in the network")
    for device in [*case.bess, *case.pv]:
        if device.bus not in case.buses:
            raise ValueError(
                f"Device {device.id!r} references bus {device.bus}, which is not in the network"
            )
    non_root = [b for b in case.buses if b != case.root]
    for b in non_root:
        if b not in case.parent_branch:
            raise ValueError(f"Bus {b} has no incoming branch (network not radial/connected?)")
    if len(case.branches) != len(non_root):
        raise ValueError(
            f"Non-radial network: {len(case.branches)} branches for {len(non_root)} "
            f"non-root buses (expected equal)."
        )
    if not case.buses[case.root].is_slack:
        raise ValueError(f"Root bus {case.root} (grid) must be of type 'slack'.")
