"""Exportação tabular das trajetórias resolvidas pelo OPF."""
from __future__ import annotations

import re
import math
from pathlib import Path

import pandas as pd

from opf.components import Case


def _column_token(value: object) -> str:
    """Converte identificadores de rede em nomes de coluna estáveis."""
    token = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    return token.strip("_").lower()


def _time_frame(case: Case) -> pd.DataFrame:
    return pd.DataFrame({
        "timestep": range(case.n_periods),
        "timestamp": case.index,
    })


def _line_current_a(
    p_kw: pd.Series,
    q_kvar: pd.Series,
    v_pu: pd.Series,
    v_base_ll_kv: float,
) -> pd.Series:
    apparent_kva = (p_kw.pow(2) + q_kvar.pow(2)).pow(0.5)
    voltage_ll_kv = v_base_ll_kv * v_pu.clip(lower=1e-9)
    return apparent_kva / (math.sqrt(3.0) * voltage_ll_kv)


def timeseries_dataframe(case: Case) -> pd.DataFrame:
    """Retorna estados da rede e resultados dos dispositivos em formato largo."""
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
            frame[f"branch_{token}_socp_gap_normalized"] = (
                result.socp_gap_normalized
            )

    for bess in sorted(case.bess, key=lambda device: device.id):
        token = _column_token(bess.id)
        result = bess.result
        frame[f"bess_{token}_p_charge_kw"] = result.charge_kw
        frame[f"bess_{token}_p_discharge_kw"] = result.discharge_kw
        frame[f"bess_{token}_p_net_kw"] = result.p_net_kw
        frame[f"bess_{token}_p_injection_kw"] = (
            result.discharge_kw - result.charge_kw
        )
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
    """Retorna uma linha por timestep e colunas por dispositivo."""
    if case.summary is None:
        raise ValueError("The case has no attached results; solve it before exporting")

    frame = _time_frame(case)
    for bess in sorted(case.bess, key=lambda device: device.id):
        token = f"bess_{_column_token(bess.id)}"
        result = bess.result
        frame[f"{token}_p_net_kw"] = (
            result.charge_kw - result.discharge_kw
        ).to_numpy()
        frame[f"{token}_q_injection_kvar"] = result.q_kvar.to_numpy()
        frame[f"{token}_p_charge_kw"] = result.charge_kw.to_numpy()
        frame[f"{token}_p_discharge_kw"] = result.discharge_kw.to_numpy()
        frame[f"{token}_soc_kwh"] = result.soc_kwh.to_numpy()
        frame[f"{token}_soc_frac"] = result.soc_frac.to_numpy()
        frame[f"{token}_inverter_loss_kw"] = (
            result.inverter_loss_kw.to_numpy()
        )

    for pv in sorted(case.pv, key=lambda device: device.id):
        token = f"pv_{_column_token(pv.id)}"
        result = pv.result
        frame[f"{token}_p_net_kw"] = (-result.p_net_kw).to_numpy()
        frame[f"{token}_q_injection_kvar"] = result.q_kvar.to_numpy()
        frame[f"{token}_available_kw"] = result.avail_kw.to_numpy()
        frame[f"{token}_generation_kw"] = result.gen_kw.to_numpy()
        frame[f"{token}_curtailment_kw"] = result.curtail_kw.to_numpy()
        frame[f"{token}_inverter_loss_kw"] = (
            result.inverter_loss_kw.to_numpy()
        )
    return frame


def buses_dataframe(case: Case) -> pd.DataFrame:
    """Retorna uma linha por timestep e colunas de estado por barra."""
    if case.summary is None:
        raise ValueError("The case has no attached results; solve it before exporting")

    frame = _time_frame(case)
    frame["price_per_kwh"] = case.price.to_numpy()
    frame["grid_import_kw"] = case.grid.result.import_kw.to_numpy()
    frame["grid_export_kw"] = case.grid.result.export_kw.to_numpy()
    frame["grid_q_kvar"] = case.grid.result.q_kvar.to_numpy()
    frame["grid_cost"] = case.grid.result.cost.to_numpy()

    for bus_id in sorted(case.buses):
        bus = case.buses[bus_id]
        token = _column_token(bus.name)
        kv_ln = float(bus.kv_base_ln or 0.0)
        v_base_ll_kv = (
            math.sqrt(3.0) * kv_ln
            if kv_ln > 0.0
            else case.base.v_base_kv
        )
        load_current = _line_current_a(
            bus.p_load_kw,
            bus.q_load_kw,
            bus.result.v_pu,
            v_base_ll_kv,
        )
        frame[f"{token}_voltage_pu"] = bus.result.v_pu.to_numpy()
        frame[f"{token}_p_load_kw"] = bus.p_load_kw.to_numpy()
        frame[f"{token}_q_load_kvar"] = bus.q_load_kw.to_numpy()
        frame[f"{token}_load_current_a"] = load_current.to_numpy()
    return frame


def branches_dataframe(case: Case) -> pd.DataFrame:
    """Retorna uma linha por timestep e colunas de estado por ramo."""
    if case.summary is None:
        raise ValueError("The case has no attached results; solve it before exporting")

    frame = _time_frame(case)
    for branch in case.branches:
        token = _column_token(
            branch.name or f"{branch.from_bus}_to_{branch.to_bus}"
        )
        result = branch.result
        from_bus = case.buses[branch.from_bus]
        kv_ln = float(from_bus.kv_base_ln or 0.0)
        v_base_ll_kv = (
            math.sqrt(3.0) * kv_ln
            if kv_ln > 0.0
            else case.base.v_base_kv
        )
        current = _line_current_a(
            result.p_kw,
            result.q_kvar,
            from_bus.result.v_pu,
            v_base_ll_kv,
        )
        frame[f"{token}_p_kw"] = result.p_kw.to_numpy()
        frame[f"{token}_q_kvar"] = result.q_kvar.to_numpy()
        frame[f"{token}_current_a"] = current.to_numpy()
        frame[f"{token}_loss_kw"] = result.loss_kw.to_numpy()
        if result.socp_gap_normalized is not None:
            frame[f"{token}_socp_gap_normalized"] = (
                result.socp_gap_normalized.to_numpy()
            )
    return frame


def summary_dataframe(case: Case) -> pd.DataFrame:
    """Retorna o resumo escalar da solução em uma linha."""
    if case.summary is None:
        raise ValueError("The case has no attached results; solve it before exporting")
    values = case.summary.as_dict()
    formulation = getattr(case, "formulation", None)
    if formulation is not None:
        values = {"formulation": formulation, **values}
    return pd.DataFrame([values]).dropna(axis="columns", how="all")


def write_result_csvs(
    case: Case,
    output_dir: str | Path,
    stem: str | None = None,
) -> dict[str, Path]:
    """Escreve devices, buses, branches e summary em CSVs separados."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    file_stem = stem or case.name
    paths = {
        "devices": destination / f"{file_stem}_devices.csv",
        "buses": destination / f"{file_stem}_buses.csv",
        "branches": destination / f"{file_stem}_branches.csv",
        "summary": destination / f"{file_stem}_summary.csv",
    }
    devices_dataframe(case).to_csv(paths["devices"], index=False)
    buses_dataframe(case).to_csv(paths["buses"], index=False)
    branches_dataframe(case).to_csv(paths["branches"], index=False)
    summary_dataframe(case).to_csv(paths["summary"], index=False)
    return paths
