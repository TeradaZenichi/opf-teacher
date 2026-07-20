"""Estruturas de dados do caso e dos resultados."""
import math

import pandas as pd


class Base:
    """Bases do sistema e conversões para pu."""

    def __init__(self, s_base_kva, v_base_kv):
        self.s_base_kva, self.v_base_kv = s_base_kva, v_base_kv
        self.z_base_ohm = v_base_kv ** 2 * 1e3 / s_base_kva

    def pu_power(self, kw): return kw / self.s_base_kva
    def pu_energy(self, kwh): return kwh / self.s_base_kva
    def pu_impedance(self, ohm): return ohm / self.z_base_ohm
    def to_kw(self, pu): return pu * self.s_base_kva
    def to_kwh(self, pu): return pu * self.s_base_kva


class BusResult:
    def __init__(self, v_pu): self.v_pu = v_pu


class BranchResult:
    def __init__(self, p_kw, q_kvar, loss_kw, socp_gap_pu2, socp_gap_normalized,
                 socp_gap_relative_flow=None, socp_current_error_a=None,
                 socp_loss_error_w=None):
        self.p_kw, self.q_kvar, self.loss_kw = p_kw, q_kvar, loss_kw
        self.socp_gap_pu2 = socp_gap_pu2
        self.socp_gap_normalized = socp_gap_normalized
        self.socp_gap_relative_flow = socp_gap_relative_flow
        self.socp_current_error_a = socp_current_error_a
        self.socp_loss_error_w = socp_loss_error_w


class GridResult:
    def __init__(self, import_kw, export_kw, q_kvar, cost):
        self.import_kw, self.export_kw, self.q_kvar, self.cost = import_kw, export_kw, q_kvar, cost


class BessResult:
    def __init__(self, charge_kw, discharge_kw, p_net_kw, soc_kwh, soc_frac,
                 q_kvar=None, inverter_loss_kw=None):
        self.charge_kw, self.discharge_kw, self.p_net_kw = charge_kw, discharge_kw, p_net_kw
        self.q_kvar = q_kvar
        self.inverter_loss_kw = inverter_loss_kw
        self.soc_kwh, self.soc_frac = soc_kwh, soc_frac


class PvResult:
    def __init__(self, avail_kw, gen_kw, curtail_kw, q_kvar,
                 inverter_loss_kw=None):
        self.avail_kw, self.gen_kw, self.curtail_kw = avail_kw, gen_kw, curtail_kw
        self.q_kvar = q_kvar
        self.inverter_loss_kw = inverter_loss_kw


class Bus:
    def __init__(self, id, type, name, v_min_pu, v_max_pu, p_load_kw, q_load_kw,
                 phases=None, kv_base_ln=None):
        self.id, self.type, self.name = id, type, name
        self.v_min_pu, self.v_max_pu = v_min_pu, v_max_pu
        self.p_load_kw, self.q_load_kw = p_load_kw, q_load_kw
        self.phases = tuple(phases or ())
        self.kv_base_ln = kv_base_ln
        self.result = None

    @property
    def is_slack(self): return self.type == "slack"


class Branch:
    def __init__(self, from_bus, to_bus, r_ohm, x_ohm, s_max_kva,
                 name=None, phases=None, norm_amps=None,
                 r_matrix_ohm=None, x_matrix_ohm=None,
                 length=None, length_units=None, element_type="line",
                 tap_ratio=1.0, r_pu_on_rating=None, x_pu_on_rating=None,
                 impedance_base_kva=None, connections=None):
        self.from_bus, self.to_bus = from_bus, to_bus
        self.r_ohm, self.x_ohm, self.s_max_kva = r_ohm, x_ohm, s_max_kva
        self.name = name
        self.phases = tuple(phases or ())
        self.norm_amps = norm_amps
        self.r_matrix_ohm = r_matrix_ohm
        self.x_matrix_ohm = x_matrix_ohm
        self.length = length
        self.length_units = length_units
        self.element_type = element_type
        self.tap_ratio = tap_ratio
        self.r_pu_on_rating = r_pu_on_rating
        self.x_pu_on_rating = x_pu_on_rating
        self.impedance_base_kva = impedance_base_kva
        self.connections = tuple(connections or ())
        self.result = None

    def impedance_pu(self, base, from_bus):
        """Retorna R e X nas bases do ramo."""
        if self.impedance_base_kva is not None:
            scale = base.s_base_kva / self.impedance_base_kva
            return self.r_pu_on_rating * scale, self.x_pu_on_rating * scale

        kv_ln = float(from_bus.kv_base_ln or 0.0)
        v_base_ll_kv = math.sqrt(3.0) * kv_ln if kv_ln > 0.0 else base.v_base_kv
        z_base_ohm = v_base_ll_kv ** 2 * 1e3 / base.s_base_kva
        return self.r_ohm / z_base_ohm, self.x_ohm / z_base_ohm


class Grid:
    def __init__(self, bus, v_ref_pu, p_import_max_kw, p_export_max_kw, q_max_kvar, feed_in_ratio):
        self.bus, self.v_ref_pu = bus, v_ref_pu
        self.p_import_max_kw, self.p_export_max_kw = p_import_max_kw, p_export_max_kw
        self.q_max_kvar, self.feed_in_ratio = q_max_kvar, feed_in_ratio
        self.result = None


class Bess:
    def __init__(self, id, bus, e_cap_kwh, p_charge_max_kw, p_discharge_max_kw,
                 eta_charge, eta_discharge, soc_init_frac, soc_min_frac, soc_max_frac,
                 cyclic_soc, s_max_kva=None, reactive_control=False,
                 q_loss_rated_kw=0.0):
        self.id, self.bus, self.e_cap_kwh = id, bus, e_cap_kwh
        self.p_charge_max_kw, self.p_discharge_max_kw = p_charge_max_kw, p_discharge_max_kw
        self.eta_charge, self.eta_discharge = eta_charge, eta_discharge
        self.soc_init_frac, self.soc_min_frac, self.soc_max_frac = soc_init_frac, soc_min_frac, soc_max_frac
        self.cyclic_soc = cyclic_soc
        self.s_max_kva = (max(p_charge_max_kw, p_discharge_max_kw)
                          if s_max_kva is None else s_max_kva)
        self.reactive_control = reactive_control
        self.q_loss_rated_kw = q_loss_rated_kw
        self.result = None


class Pv:
    def __init__(self, id, bus, p_max_kw, s_max_kva, control, curtailable,
                 power_factor, avail_kw, q_loss_rated_kw=0.0):
        self.id, self.bus, self.p_max_kw, self.s_max_kva = id, bus, p_max_kw, s_max_kva
        self.control, self.curtailable, self.power_factor = control, curtailable, power_factor
        self.avail_kw = avail_kw
        self.q_loss_rated_kw = q_loss_rated_kw
        self.result = None


class Summary:
    def __init__(self, status, objective_cost, energy_cost, energy_import_kwh, energy_export_kwh,
                 pv_generated_kwh, pv_curtailed_kwh, losses_kwh, v_min_pu, v_max_pu,
                 socp_gap_max_pu2, socp_gap_max_normalized, socp_gap_tolerance,
                 socp_tightness_margin, socp_tightness, socp_relaxation_tight,
                 inverter_losses_kwh=0.0, socp_gap_max_relative_flow=0.0,
                 socp_current_error_max_a=0.0, socp_loss_error_max_w=0.0):
        self.status, self.objective_cost, self.energy_cost = status, objective_cost, energy_cost
        self.energy_import_kwh, self.energy_export_kwh = energy_import_kwh, energy_export_kwh
        self.pv_generated_kwh, self.pv_curtailed_kwh = pv_generated_kwh, pv_curtailed_kwh
        self.losses_kwh = losses_kwh
        self.v_min_pu, self.v_max_pu = v_min_pu, v_max_pu
        self.socp_gap_max_pu2 = socp_gap_max_pu2
        self.socp_gap_max_normalized = socp_gap_max_normalized
        self.socp_gap_tolerance = socp_gap_tolerance
        self.socp_tightness_margin = socp_tightness_margin
        self.socp_tightness = socp_tightness
        self.socp_relaxation_tight = socp_relaxation_tight
        self.inverter_losses_kwh = inverter_losses_kwh
        self.socp_gap_max_relative_flow = socp_gap_max_relative_flow
        self.socp_current_error_max_a = socp_current_error_max_a
        self.socp_loss_error_max_w = socp_loss_error_max_w

    @property
    def socp_confidence_margin(self):
        """Alias mantido para consumidores da API anterior."""
        return self.socp_tightness_margin

    @property
    def socp_confidence(self):
        """Alias mantido para consumidores da API anterior."""
        return self.socp_tightness

    def as_dict(self): return dict(self.__dict__)


class Case:
    def __init__(self, name, base, buses, branches, grid, bess, pv,
                 timestamps, dt_h, price):
        self.name, self.base = name, base
        self.buses, self.branches, self.grid = buses, branches, grid
        self.bess, self.pv = bess, pv
        self.timestamps, self.dt_h, self.price = timestamps, dt_h, price
        self.parent_branch, self.children = {}, {}
        self.bus_name_to_id = {}
        self.summary = None

    @property
    def periods(self): return range(len(self.timestamps))
    @property
    def n_periods(self): return len(self.timestamps)
    @property
    def root(self): return self.grid.bus
    @property
    def index(self): return pd.DatetimeIndex(self.timestamps)
