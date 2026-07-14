"""Read an example (config.json + devices.json + CSVs) into a Case.

Everything stays in physical units (kW/kVAr/kWh/ohm/kV) -- the model does the
per-unit conversion. The demand.csv timestamps define the time axis (periods, dt).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from opf.components import Base, Bess, Branch, Bus, Case, Grid, Pv


def load_case(path: str | Path) -> Case:
    path = Path(path)
    cfg = _read_json(path / "config.json")

    base = Base(float(cfg["base"]["s_base_kva"]), float(cfg["base"]["v_base_kv"]))
    vlim = cfg.get("voltage_limits", {})
    v_min = float(vlim.get("v_min_pu", 0.95))
    v_max = float(vlim.get("v_max_pu", 1.05))
    pf_default = float(cfg.get("defaults", {}).get("power_factor", 1.0))

    # demand.csv is wide (P2,Q2,P3,Q3,...); split each column into a per-bus series
    dem = (pd.read_csv(path / "demand.csv", parse_dates=["timestamp"])
           .sort_values("timestamp").reset_index(drop=True))
    idx = pd.DatetimeIndex(dem["timestamp"])
    p_load, q_load = {}, {}
    for col in dem.columns:
        if col == "timestamp":
            continue
        kind, bus = col[0].upper(), int(col[1:])     # "P2" -> ("P", 2)
        (p_load if kind == "P" else q_load)[bus] = pd.Series(dem[col].to_numpy(), index=idx)

    zero = pd.Series(0.0, index=idx)
    q_ratio = math.tan(math.acos(pf_default))

    bus_df = pd.read_csv(path / "bus.csv")
    has_vmin, has_vmax = "v_min_pu" in bus_df.columns, "v_max_pu" in bus_df.columns
    buses: dict[int, Bus] = {}
    for _, r in bus_df.iterrows():
        bid = int(r["bus_id"])
        p = p_load.get(bid, zero.copy())
        q = q_load.get(bid, p * q_ratio if bid in p_load else zero.copy())  # Q from pf if not given
        buses[bid] = Bus(
            id=bid,
            type=str(r["type"]).strip().lower(),
            name=str(r.get("name", f"B{bid}")),
            v_min_pu=float(r["v_min_pu"]) if has_vmin else v_min,
            v_max_pu=float(r["v_max_pu"]) if has_vmax else v_max,
            p_load_kw=p,
            q_load_kw=q,
        )

    br_df = pd.read_csv(path / "branch.csv")
    branches = [
        Branch(int(r["from_bus"]), int(r["to_bus"]),
               float(r["r_ohm"]), float(r["x_ohm"]), float(r["s_max_kva"]))
        for _, r in br_df.iterrows()
    ]

    g = cfg["grid"]
    grid = Grid(
        bus=int(g["bus"]),
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
            id=str(d["id"]), bus=int(d["bus"]),
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
            id=str(d["id"]), bus=int(d["bus"]),
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
        evse=list(dev.get("evse", [])),
        timestamps=list(idx),
        dt_h=_infer_dt_hours(idx),
        price=price,
        objective=cfg.get("objective", {"minimize": "energy_cost"}),
    )
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
    """Read a 'file.csv:column' profile, aligned to the time axis."""
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
