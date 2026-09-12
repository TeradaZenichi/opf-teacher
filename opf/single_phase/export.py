"""Export solved single-phase trajectories as tables and CSV files."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from opf.components import (Case, phase_power_factor, system_voltage_base_kv)


def _column_token(value: object) -> str:
    token = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    return token.strip("_").lower()


def _time_frame(case: Case) -> pd.DataFrame:
    if case.summary is None:
        raise ValueError("The case has no attached results; solve it before exporting")
    return pd.DataFrame({"timestep": range(case.n_periods), "timestamp": case.index})


def _line_current_a(p_kw: pd.Series, q_kvar: pd.Series, v_pu: pd.Series, voltage_base_kv: float, phases) -> pd.Series:
    apparent_kva = (p_kw.pow(2) + q_kvar.pow(2)).pow(0.5)
    voltage_kv = voltage_base_kv * v_pu.clip(lower=1e-9)
    return apparent_kva / (phase_power_factor(phases) * voltage_kv)


def timeseries_dataframe(case: Case) -> pd.DataFrame:
    """Combine network and device results in a wide time-series table."""
    if case.summary is None:
        raise ValueError("The case has no attached results; solve it before exporting")

    frame = pd.DataFrame(index=case.index)
    frame.index.name = "timestamp"
    frame["price_per_kwh"] = case.price
    frame["grid_import_kw"] = case.grid.result.import_kw
    frame["grid_export_kw"] = case.grid.result.export_kw
    frame["grid_q_kvar"] = case.grid.result.q_kvar
    frame["grid_cost"] = case.grid.result.cost

    for bus_id in sorted(case.buses):
        bus = case.buses[bus_id]
        token = _column_token(bus.name)
        frame[f"bus_{token}_p_load_kw"] = bus.p_load_kw
        frame[f"bus_{token}_q_load_kvar"] = bus.q_load_kw
        frame[f"bus_{token}_v_pu"] = bus.result.v_pu

    for branch in case.branches:
        name = branch.name or f"{branch.from_bus}_to_{branch.to_bus}"
        token = _column_token(name)
        result = branch.result
        frame[f"branch_{token}_p_kw"] = result.p_kw
        frame[f"branch_{token}_q_kvar"] = result.q_kvar
        frame[f"branch_{token}_loss_kw"] = result.loss_kw
        if result.socp_gap_normalized is not None:
            frame[f"branch_{token}_socp_gap_normalized"] = (result.socp_gap_normalized)

    for bess in sorted(case.bess, key=lambda device: device.id):
        token = _column_token(bess.id)
        result = bess.result
        frame[f"bess_{token}_p_charge_kw"] = result.charge_kw
        frame[f"bess_{token}_p_discharge_kw"] = result.discharge_kw
        frame[f"bess_{token}_p_net_kw"] = result.p_net_kw
        frame[f"bess_{token}_p_injection_kw"] = (result.discharge_kw - result.charge_kw)
        frame[f"bess_{token}_q_kvar"] = result.q_kvar
        frame[f"bess_{token}_soc_kwh"] = result.soc_kwh
        frame[f"bess_{token}_soc_frac"] = result.soc_frac
        frame[f"bess_{token}_inverter_loss_kw"] = result.inverter_loss_kw

    for pv in sorted(case.pv, key=lambda device: device.id):
        token = _column_token(pv.id)
        result = pv.result
        frame[f"pv_{token}_available_kw"] = result.avail_kw
        frame[f"pv_{token}_generation_kw"] = result.gen_kw
        frame[f"pv_{token}_curtailment_kw"] = result.curtail_kw
        frame[f"pv_{token}_p_net_kw"] = result.p_net_kw
        frame[f"pv_{token}_q_kvar"] = result.q_kvar
        frame[f"pv_{token}_inverter_loss_kw"] = result.inverter_loss_kw

    return frame.reset_index()


def devices_dataframe(case: Case) -> pd.DataFrame:
    frame = _time_frame(case)
    for bess in sorted(case.bess, key=lambda device: device.id):
        token = f"bess_{_column_token(bess.id)}"
        result = bess.result
        frame[f"{token}_p_net_kw"] = (result.charge_kw - result.discharge_kw).to_numpy()
        frame[f"{token}_q_injection_kvar"] = result.q_kvar.to_numpy()
        frame[f"{token}_p_charge_kw"] = result.charge_kw.to_numpy()
        frame[f"{token}_p_discharge_kw"] = result.discharge_kw.to_numpy()
        frame[f"{token}_soc_kwh"] = result.soc_kwh.to_numpy()
        frame[f"{token}_soc_frac"] = result.soc_frac.to_numpy()
        frame[f"{token}_inverter_loss_kw"] = (result.inverter_loss_kw.to_numpy())

    for pv in sorted(case.pv, key=lambda device: device.id):
        token = f"pv_{_column_token(pv.id)}"
        result = pv.result
        frame[f"{token}_p_net_kw"] = (-result.p_net_kw).to_numpy()
        frame[f"{token}_q_injection_kvar"] = result.q_kvar.to_numpy()
        frame[f"{token}_available_kw"] = result.avail_kw.to_numpy()
        frame[f"{token}_generation_kw"] = result.gen_kw.to_numpy()
        frame[f"{token}_curtailment_kw"] = result.curtail_kw.to_numpy()
        frame[f"{token}_inverter_loss_kw"] = (result.inverter_loss_kw.to_numpy())
    return frame


def buses_dataframe(case: Case) -> pd.DataFrame:
    frame = _time_frame(case)
    frame["price_per_kwh"] = case.price.to_numpy()
    frame["grid_import_kw"] = case.grid.result.import_kw.to_numpy()
    frame["grid_export_kw"] = case.grid.result.export_kw.to_numpy()
    frame["grid_q_kvar"] = case.grid.result.q_kvar.to_numpy()
    frame["grid_cost"] = case.grid.result.cost.to_numpy()

    for bus_id in sorted(case.buses):
        bus = case.buses[bus_id]
        token = _column_token(bus.name)
        voltage_base_kv = system_voltage_base_kv(bus.kv_base_ln, bus.phases, case.base.v_base_kv)
        load_current = _line_current_a(bus.p_load_kw, bus.q_load_kw, bus.result.v_pu, voltage_base_kv, bus.phases)
        frame[f"{token}_voltage_pu"] = bus.result.v_pu.to_numpy()
        frame[f"{token}_p_load_kw"] = bus.p_load_kw.to_numpy()
        frame[f"{token}_q_load_kvar"] = bus.q_load_kw.to_numpy()
        frame[f"{token}_load_current_a"] = load_current.to_numpy()
    return frame


def branches_dataframe(case: Case) -> pd.DataFrame:
    frame = _time_frame(case)
    for branch in case.branches:
        token = _column_token(branch.name or f"{branch.from_bus}_to_{branch.to_bus}")
        result = branch.result
        from_bus = case.buses[branch.from_bus]
        voltage_base_kv = system_voltage_base_kv(from_bus.kv_base_ln, branch.phases, case.base.v_base_kv)
        current = _line_current_a(result.p_kw, result.q_kvar, from_bus.result.v_pu, voltage_base_kv, branch.phases)
        frame[f"{token}_p_kw"] = result.p_kw.to_numpy()
        frame[f"{token}_q_kvar"] = result.q_kvar.to_numpy()
        frame[f"{token}_current_a"] = current.to_numpy()
        frame[f"{token}_loss_kw"] = result.loss_kw.to_numpy()
        if result.socp_gap_normalized is not None:
            frame[f"{token}_socp_gap_normalized"] = (result.socp_gap_normalized.to_numpy())
    return frame


def summary_dataframe(case: Case) -> pd.DataFrame:
    if case.summary is None:
        raise ValueError("The case has no attached results; solve it before exporting")
    values = case.summary.as_dict()
    formulation = getattr(case, "formulation", None)
    if formulation is not None:
        values = {"formulation": formulation, **values}
    return pd.DataFrame([values]).dropna(axis="columns", how="all")


def write_result_csvs(case: Case, output_dir: str | Path, stem: str | None = None) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    file_stem = stem or case.name
    exporters = {
        "devices": devices_dataframe,
        "buses": buses_dataframe,
        "branches": branches_dataframe,
        "summary": summary_dataframe,
    }
    paths = {}
    for name, to_frame in exporters.items():
        paths[name] = destination / f"{file_stem}_{name}.csv"
        to_frame(case).to_csv(paths[name], index=False)
    return paths
