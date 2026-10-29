"""Phase-indexed result contracts for the unbalanced teacher."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

import pandas as pd

from opf.common import Summary
from opf.three_phase.components import Case, _profile_map
from opf.three_phase.phases import normalize_phases


if TYPE_CHECKING:
    from opf.three_phase.ivr_model import IVRModel


def _aligned_profiles(values, phases, *, label, index=None):
    profiles = _profile_map(values, phases, label=label)
    profile_index = next(iter(profiles.values())).index
    if index is not None and not profile_index.equals(index):
        raise ValueError(f"{label} does not match the expected result horizon")
    return profiles, profile_index


@dataclass
class BusResult:
    v_pu: Mapping[str | int, pd.Series]
    angle_deg: Mapping[str | int, pd.Series]

    def __post_init__(self):
        phases = normalize_phases(self.v_pu)
        self.v_pu, index = _aligned_profiles(
            self.v_pu, phases, label="bus voltage result"
        )
        self.angle_deg, _ = _aligned_profiles(
            self.angle_deg, phases, label="bus voltage angle result", index=index
        )
        if any((series <= 0.0).any() for series in self.v_pu.values()):
            raise ValueError("Bus voltage result must be positive")


@dataclass
class BranchResult:
    p_kw: Mapping[str | int, pd.Series]
    q_kvar: Mapping[str | int, pd.Series]
    current_a: Mapping[str | int, pd.Series]
    loss_kw: Mapping[str | int, pd.Series]

    def __post_init__(self):
        phases = normalize_phases(self.p_kw)
        self.p_kw, index = _aligned_profiles(
            self.p_kw, phases, label="branch active flow result"
        )
        self.q_kvar, _ = _aligned_profiles(
            self.q_kvar, phases, label="branch reactive flow result", index=index
        )
        self.current_a, _ = _aligned_profiles(
            self.current_a, phases, label="branch current result", index=index
        )
        self.loss_kw, _ = _aligned_profiles(
            self.loss_kw, phases, label="branch loss result", index=index
        )
        if any((series < 0.0).any() for series in self.current_a.values()):
            raise ValueError("Branch current magnitude cannot be negative")


@dataclass
class GridResult:
    import_kw: Mapping[str | int, pd.Series]
    export_kw: Mapping[str | int, pd.Series]
    q_kvar: Mapping[str | int, pd.Series]
    cost: pd.Series

    def __post_init__(self):
        phases = normalize_phases(self.import_kw)
        self.import_kw, index = _aligned_profiles(
            self.import_kw, phases, label="grid import result"
        )
        self.export_kw, _ = _aligned_profiles(
            self.export_kw, phases, label="grid export result", index=index
        )
        self.q_kvar, _ = _aligned_profiles(
            self.q_kvar, phases, label="grid reactive result", index=index
        )
        self.cost = self.cost.astype(float).copy()
        if not self.cost.index.equals(index):
            raise ValueError("Grid cost does not match the result horizon")
        if self.cost.isna().any() or not self.cost.map(math.isfinite).all():
            raise ValueError("Grid cost must contain finite values")
        if any((series < 0.0).any() for series in (*self.import_kw.values(), *self.export_kw.values())):
            raise ValueError("Grid import and export cannot be negative")

    @property
    def import_total_kw(self):
        index = next(iter(self.import_kw.values())).index
        return sum(self.import_kw.values(), start=pd.Series(0.0, index=index))

    @property
    def export_total_kw(self):
        index = next(iter(self.export_kw.values())).index
        return sum(self.export_kw.values(), start=pd.Series(0.0, index=index))

    @property
    def q_total_kvar(self):
        index = next(iter(self.q_kvar.values())).index
        return sum(self.q_kvar.values(), start=pd.Series(0.0, index=index))


@dataclass
class BessResult:
    p_net_kw: Mapping[str | int, pd.Series]
    q_injection_kvar: Mapping[str | int, pd.Series]
    soc_kwh: pd.Series
    soc_frac: pd.Series

    def __post_init__(self):
        phases = normalize_phases(self.p_net_kw)
        self.p_net_kw, index = _aligned_profiles(
            self.p_net_kw, phases, label="BESS active result"
        )
        self.q_injection_kvar, _ = _aligned_profiles(
            self.q_injection_kvar, phases, label="BESS reactive result", index=index
        )
        self.soc_kwh = self.soc_kwh.astype(float).copy()
        self.soc_frac = self.soc_frac.astype(float).copy()
        if not self.soc_kwh.index.equals(index) or not self.soc_frac.index.equals(index):
            raise ValueError("BESS SoC results do not match the phase-result horizon")
        tolerance = 1e-7
        if (self.soc_kwh < -tolerance).any() or (
            (self.soc_frac < -tolerance) | (self.soc_frac > 1.0 + tolerance)
        ).any():
            raise ValueError("Invalid BESS SoC result")
        self.soc_kwh = self.soc_kwh.clip(lower=0.0)
        self.soc_frac = self.soc_frac.clip(lower=0.0, upper=1.0)

    @property
    def p_net_total_kw(self):
        index = next(iter(self.p_net_kw.values())).index
        return sum(self.p_net_kw.values(), start=pd.Series(0.0, index=index))

    @property
    def q_injection_total_kvar(self):
        index = next(iter(self.q_injection_kvar.values())).index
        return sum(
            self.q_injection_kvar.values(), start=pd.Series(0.0, index=index)
        )


@dataclass
class PvResult:
    available_kw: Mapping[str | int, pd.Series]
    generation_kw: Mapping[str | int, pd.Series]
    q_injection_kvar: Mapping[str | int, pd.Series]
    curtailment_kw: Mapping[str | int, pd.Series]

    def __post_init__(self):
        phases = normalize_phases(self.available_kw)
        self.available_kw, index = _aligned_profiles(
            self.available_kw, phases, label="PV availability result"
        )
        self.generation_kw, _ = _aligned_profiles(
            self.generation_kw, phases, label="PV generation result", index=index
        )
        self.q_injection_kvar, _ = _aligned_profiles(
            self.q_injection_kvar, phases, label="PV reactive result", index=index
        )
        self.curtailment_kw, _ = _aligned_profiles(
            self.curtailment_kw, phases, label="PV curtailment result", index=index
        )
        nonnegative = (
            *self.available_kw.values(),
            *self.generation_kw.values(),
            *self.curtailment_kw.values(),
        )
        if any((series < 0.0).any() for series in nonnegative):
            raise ValueError("PV availability, generation and curtailment cannot be negative")

    @property
    def generation_total_kw(self):
        index = next(iter(self.generation_kw.values())).index
        return sum(self.generation_kw.values(), start=pd.Series(0.0, index=index))

    @property
    def q_injection_total_kvar(self):
        index = next(iter(self.q_injection_kvar.values())).index
        return sum(
            self.q_injection_kvar.values(), start=pd.Series(0.0, index=index)
        )

    @property
    def curtailment_total_kw(self):
        index = next(iter(self.curtailment_kw.values())).index
        return sum(
            self.curtailment_kw.values(), start=pd.Series(0.0, index=index)
        )


def attach_results(model: IVRModel, case: Case) -> Case:
    """Attach physical results by component, then calculate the summary."""
    if model.solution is None:
        raise ValueError("The three-phase IVR model has not been solved")
    _attach_bus_results(model, case)
    network_losses = _attach_branch_results(model, case)
    _attach_grid_results(model, case)
    bess_losses = _attach_bess_results(model, case)
    pv_curtailed, pv_losses = _attach_pv_results(model, case)
    totals = EnergyTotals(network_losses, bess_losses + pv_losses, pv_curtailed)
    _attach_summary(model, case, totals)
    return case


@dataclass
class EnergyTotals:
    """Horizon sums of power samples; multiply by dt_h to obtain energy."""

    network_losses: float
    inverter_losses: float
    pv_curtailed: float


def _phase_series(values, index):
    profiles = {}
    for phase, items in values.items():
        profiles[phase] = pd.Series(items, index=index, dtype=float)
    return profiles


def _attach_bus_results(model, case):
    solution = model.solution
    index = case.index
    for bus_id, bus in case.buses.items():
        voltage = {}
        angle = {}
        for phase in bus.phases:
            position = model.node_position[(bus_id, phase)]
            values = np.asarray([item[position] for item in solution.voltages_kv])
            voltage[phase] = pd.Series(
                abs(values) / model.voltage_bases_kv[position], index=index
            )
            angle[phase] = pd.Series(np.degrees(np.angle(values)), index=index)
        bus.result = BusResult(v_pu=voltage, angle_deg=angle)


def _attach_branch_results(model, case):
    solution = model.solution
    index = case.index
    total_losses_kw = 0.0
    for branch_index, branch in enumerate(case.branches):
        p_values = {phase: [] for phase in branch.phases}
        q_values = {phase: [] for phase in branch.phases}
        i_values = {phase: [] for phase in branch.phases}
        loss_values = {phase: [] for phase in branch.phases}
        for time, voltage in enumerate(solution.voltages_kv):
            current = solution.branch_currents_ka[time][branch_index]
            from_voltage = np.asarray([
                voltage[model.node_position[(branch.from_bus, phase)]]
                for phase in branch.phases
            ])
            to_voltage = np.asarray([
                voltage[model.node_position[(branch.to_bus, phase)]]
                for phase in branch.phases
            ])
            sending_voltage = (
                model.branch_from_connections[branch_index] @ from_voltage
            )
            receiving_voltage = (
                model.branch_to_connections[branch_index] @ to_voltage
            )
            sending = sending_voltage * np.conj(current) * 1000.0
            to_current = model.branch_taps[branch_index].conj().T @ current
            receiving = receiving_voltage * np.conj(to_current) * 1000.0
            losses = sending.real - receiving.real
            for position, phase in enumerate(branch.phases):
                p_values[phase].append(sending[position].real)
                q_values[phase].append(sending[position].imag)
                i_values[phase].append(abs(current[position]) * 1000.0)
                loss_values[phase].append(losses[position])
        branch.result = BranchResult(
            p_kw=_phase_series(p_values, index),
            q_kvar=_phase_series(q_values, index),
            current_a=_phase_series(i_values, index),
            loss_kw=_phase_series(loss_values, index),
        )
        total_losses_kw += sum(float(values.sum()) for values in branch.result.loss_kw.values())

    return total_losses_kw


def _attach_grid_results(model, case):
    solution = model.solution
    index = case.index
    grid_import = {phase: [] for phase in case.grid.phases}
    grid_export = {phase: [] for phase in case.grid.phases}
    grid_q = {phase: [] for phase in case.grid.phases}
    for values in solution.grid_power_kva:
        for position, phase in enumerate(case.grid.phases):
            grid_import[phase].append(max(values[position].real, 0.0))
            grid_export[phase].append(max(-values[position].real, 0.0))
            grid_q[phase].append(values[position].imag)
    import_series = _phase_series(grid_import, index)
    export_series = _phase_series(grid_export, index)
    cost = case.dt_h * case.price * (
        sum(import_series.values(), start=pd.Series(0.0, index=index))
        - case.grid.feed_in_ratio
        * sum(export_series.values(), start=pd.Series(0.0, index=index))
    )
    case.grid.result = GridResult(
        import_kw=import_series,
        export_kw=export_series,
        q_kvar=_phase_series(grid_q, index),
        cost=cost,
    )


def _attach_bess_results(model, case):
    solution = model.solution
    index = case.index
    inverter_losses_kw = 0.0
    for device_index, connected in enumerate(case.bess):
        battery = connected.device
        connected.result = BessResult(
            p_net_kw={
                phase: pd.Series(
                    solution.p_charge_phase_kw[
                        device_index, model.phase_position[phase]
                    ]
                    - solution.p_discharge_phase_kw[
                        device_index, model.phase_position[phase]
                    ],
                    index=index,
                )
                for phase in connected.connection.phases
            },
            q_injection_kvar={
                phase: pd.Series(
                    solution.q_bess_phase_kvar[
                        device_index, model.phase_position[phase]
                    ],
                    index=index,
                )
                for phase in connected.connection.phases
            },
            soc_kwh=pd.Series(solution.soc_kwh[device_index], index=index),
            soc_frac=pd.Series(
                solution.soc_kwh[device_index] / battery.e_cap_kwh,
                index=index,
            ),
        )
        if connected.connection.dispatch_mode == "aggregate":
            inverter_losses_kw += sum(
                model._bess_q_loss(device_index, value)
                for value in solution.q_bess_kvar[device_index]
            )
        else:
            phase_indices = [
                model.phase_position[phase]
                for phase in connected.connection.phases
            ]
            inverter_losses_kw += sum(
                model._bess_q_loss(
                    device_index,
                    solution.q_bess_phase_kvar[
                        device_index, phase_indices, time
                    ],
                )
                for time in case.periods
            )

    return inverter_losses_kw


def _attach_pv_results(model, case):
    solution = model.solution
    index = case.index
    inverter_losses_kw = 0.0
    pv_curtailed = 0.0
    for device_index, connected in enumerate(case.pv):
        generated = {}
        reactive = {}
        curtailed = {}
        for phase in connected.connection.phases:
            phase_index = model.phase_position[phase]
            available = connected.available_kw[phase].to_numpy()
            phase_generation = solution.pv_generation_phase_kw[
                device_index, phase_index
            ]
            generated[phase] = pd.Series(phase_generation, index=index)
            reactive[phase] = pd.Series(
                solution.q_pv_phase_kvar[device_index, phase_index],
                index=index,
            )
            curtailed[phase] = pd.Series(
                np.maximum(available - phase_generation, 0.0), index=index
            )
        connected.result = PvResult(
            available_kw=connected.available_kw,
            generation_kw=generated,
            q_injection_kvar=reactive,
            curtailment_kw=curtailed,
        )
        pv_curtailed += sum(float(values.sum()) for values in curtailed.values())
        if connected.connection.dispatch_mode == "aggregate":
            inverter_losses_kw += sum(
                model._pv_q_loss(device_index, value)
                for value in solution.q_pv_kvar[device_index]
            )
        else:
            phase_indices = [
                model.phase_position[phase]
                for phase in connected.connection.phases
            ]
            inverter_losses_kw += sum(
                model._pv_q_loss(
                    device_index,
                    solution.q_pv_phase_kvar[
                        device_index, phase_indices, time
                    ],
                )
                for time in case.periods
            )

    return pv_curtailed, inverter_losses_kw


def _attach_summary(model, case, totals: EnergyTotals):
    solution = model.solution
    all_voltages = [
        value
        for bus in case.buses.values()
        for series in bus.result.v_pu.values()
        for value in series
    ]
    imported = case.grid.result.import_total_kw
    exported = case.grid.result.export_total_kw
    pv_generated = sum(
        float(series.sum())
        for connected in case.pv
        for series in connected.result.generation_kw.values()
    )
    case.summary = Summary(
        status="locally_optimal",
        objective_cost=solution.objective_cost,
        energy_cost=float(case.grid.result.cost.sum()),
        energy_import_kwh=float(imported.sum()) * case.dt_h,
        energy_export_kwh=float(exported.sum()) * case.dt_h,
        pv_generated_kwh=pv_generated * case.dt_h,
        pv_curtailed_kwh=totals.pv_curtailed * case.dt_h,
        losses_kwh=totals.network_losses * case.dt_h,
        v_min_pu=min(all_voltages),
        v_max_pu=max(all_voltages),
        socp_gap_max_pu2=float("nan"),
        socp_gap_max_normalized=float("nan"),
        socp_gap_tolerance=float("nan"),
        socp_tightness_margin=float("nan"),
        socp_tightness="not_applicable",
        socp_relaxation_tight=False,
        inverter_losses_kwh=totals.inverter_losses * case.dt_h,
        socp_gap_max_relative_flow=float("nan"),
        socp_current_error_max_a=float("nan"),
        socp_loss_error_max_w=float("nan"),
    )
    case.formulation = "three_phase_ac_ivr_reduced_space"
    case.quality = {
        "optimizer": "scipy_slsqp",
        "iterations": solution.iterations,
        "max_power_flow_residual_a": solution.max_power_flow_residual_a,
        "solution_quality": "local_feasible",
    }
