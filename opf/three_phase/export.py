"""Wide CSV exports for phase-native OPF results."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from opf.three_phase.components import Case


def _token(value):
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip()).strip("_").lower()


def _frame(case):
    if case.summary is None:
        raise ValueError("The three-phase case has no attached results")
    return pd.DataFrame({"timestep": range(case.n_periods), "timestamp": case.index})


def buses_dataframe(case: Case):
    frame = _frame(case)
    frame["price_per_kwh"] = case.price.to_numpy()
    frame["grid_import_total_kw"] = case.grid.result.import_total_kw.to_numpy()
    frame["grid_export_total_kw"] = case.grid.result.export_total_kw.to_numpy()
    for phase in case.grid.phases:
        frame[f"grid_import_{phase}_kw"] = case.grid.result.import_kw[phase].to_numpy()
        frame[f"grid_export_{phase}_kw"] = case.grid.result.export_kw[phase].to_numpy()
        frame[f"grid_q_{phase}_kvar"] = case.grid.result.q_kvar[phase].to_numpy()
    for bus_id in sorted(case.buses):
        bus = case.buses[bus_id]
        token = _token(bus.name)
        for phase in bus.phases:
            frame[f"{token}_voltage_{phase}_pu"] = bus.result.v_pu[phase].to_numpy()
            frame[f"{token}_angle_{phase}_deg"] = bus.result.angle_deg[phase].to_numpy()
            frame[f"{token}_p_load_{phase}_kw"] = bus.p_load_kw[phase].to_numpy()
            frame[f"{token}_q_load_{phase}_kvar"] = bus.q_load_kvar[phase].to_numpy()
    return frame


def branches_dataframe(case: Case):
    frame = _frame(case)
    for branch in case.branches:
        token = _token(branch.name)
        for phase in branch.phases:
            frame[f"{token}_p_{phase}_kw"] = branch.result.p_kw[phase].to_numpy()
            frame[f"{token}_q_{phase}_kvar"] = branch.result.q_kvar[phase].to_numpy()
            frame[f"{token}_current_{phase}_a"] = branch.result.current_a[phase].to_numpy()
            frame[f"{token}_loss_{phase}_kw"] = branch.result.loss_kw[phase].to_numpy()
        frame[f"{token}_loss_total_kw"] = sum(
            branch.result.loss_kw.values(), start=pd.Series(0.0, index=case.index)
        ).to_numpy()
    return frame


def devices_dataframe(case: Case):
    frame = _frame(case)
    for connected in case.bess:
        token = f"bess_{_token(connected.id)}"
        for phase in connected.connection.phases:
            frame[f"{token}_p_net_{phase}_kw"] = connected.result.p_net_kw[phase].to_numpy()
            frame[f"{token}_q_injection_{phase}_kvar"] = (
                connected.result.q_injection_kvar[phase].to_numpy()
            )
        frame[f"{token}_p_net_total_kw"] = connected.result.p_net_total_kw.to_numpy()
        frame[f"{token}_soc_kwh"] = connected.result.soc_kwh.to_numpy()
        frame[f"{token}_soc_frac"] = connected.result.soc_frac.to_numpy()
    for connected in case.pv:
        token = f"pv_{_token(connected.id)}"
        for phase in connected.connection.phases:
            frame[f"{token}_available_{phase}_kw"] = connected.result.available_kw[phase].to_numpy()
            frame[f"{token}_generation_{phase}_kw"] = connected.result.generation_kw[phase].to_numpy()
            frame[f"{token}_q_injection_{phase}_kvar"] = (
                connected.result.q_injection_kvar[phase].to_numpy()
            )
        frame[f"{token}_generation_total_kw"] = connected.result.generation_total_kw.to_numpy()
    return frame


def summary_dataframe(case: Case):
    values = {"formulation": case.formulation, **case.summary.as_dict(), **case.quality}
    return pd.DataFrame([values]).dropna(axis="columns", how="all")


def write_result_csvs(case: Case, output_dir: str | Path, stem: str | None = None):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = stem or case.name
    exporters = {
        "devices": devices_dataframe,
        "buses": buses_dataframe,
        "branches": branches_dataframe,
        "summary": summary_dataframe,
    }
    paths = {}
    for name, exporter in exporters.items():
        path = destination / f"{stem}_{name}.csv"
        exporter(case).to_csv(path, index=False)
        paths[name] = path
    return paths
