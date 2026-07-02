"""Classes de dominio do problema (parametros + resultados).

Parametros em unidades fisicas (kW, kVAr, kWh, ohm, kV). A conversao para
por-unidade fica na classe Base e e usada dentro do modelo. Apos o solve, cada
componente recebe um objeto `.result` com as series temporais da solucao:

    case.bess[0].result.soc_kwh
    case.grid.result.import_kw
    case.buses[2].result.v_pu
"""
import pandas as pd


class Base:
    """Bases do sistema e conversoes fisico <-> por-unidade (Z_base = V^2/S)."""

    def __init__(self, s_base_kva, v_base_kv):
        self.s_base_kva, self.v_base_kv = s_base_kva, v_base_kv
        self.z_base_ohm = v_base_kv ** 2 * 1e3 / s_base_kva

    def pu_power(self, kw): return kw / self.s_base_kva         # kW/kVAr/kVA -> p.u.
    def pu_energy(self, kwh): return kwh / self.s_base_kva      # kWh -> p.u.*h
    def pu_impedance(self, ohm): return ohm / self.z_base_ohm   # ohm -> p.u.
    def to_kw(self, pu): return pu * self.s_base_kva
    def to_kwh(self, pu): return pu * self.s_base_kva


# --- resultados (preenchidos apos o solve) ------------------------------- #
class BusResult:
    def __init__(self, v_pu): self.v_pu = v_pu


class BranchResult:
    def __init__(self, p_kw, q_kvar, loss_kw):
        self.p_kw, self.q_kvar, self.loss_kw = p_kw, q_kvar, loss_kw


class GridResult:
    def __init__(self, import_kw, export_kw, q_kvar, cost):
        self.import_kw, self.export_kw, self.q_kvar, self.cost = import_kw, export_kw, q_kvar, cost


class BessResult:
    def __init__(self, charge_kw, discharge_kw, p_net_kw, soc_kwh, soc_frac):
        self.charge_kw, self.discharge_kw, self.p_net_kw = charge_kw, discharge_kw, p_net_kw
        self.soc_kwh, self.soc_frac = soc_kwh, soc_frac


class PvResult:
    def __init__(self, avail_kw, gen_kw, curtail_kw):
        self.avail_kw, self.gen_kw, self.curtail_kw = avail_kw, gen_kw, curtail_kw


# --- componentes (parametros fisicos + .result) -------------------------- #
class Bus:
    def __init__(self, id, type, name, v_min_pu, v_max_pu, p_load_kw, q_load_kw):
        self.id, self.type, self.name = id, type, name
        self.v_min_pu, self.v_max_pu = v_min_pu, v_max_pu
        self.p_load_kw, self.q_load_kw = p_load_kw, q_load_kw
        self.result = None

    @property
    def is_slack(self): return self.type == "slack"


class Branch:
    def __init__(self, from_bus, to_bus, r_ohm, x_ohm, s_max_kva):
        self.from_bus, self.to_bus = from_bus, to_bus
        self.r_ohm, self.x_ohm, self.s_max_kva = r_ohm, x_ohm, s_max_kva
        self.result = None


class Grid:
    def __init__(self, bus, v_ref_pu, p_import_max_kw, p_export_max_kw, q_max_kvar, feed_in_ratio):
        self.bus, self.v_ref_pu = bus, v_ref_pu
        self.p_import_max_kw, self.p_export_max_kw = p_import_max_kw, p_export_max_kw
        self.q_max_kvar, self.feed_in_ratio = q_max_kvar, feed_in_ratio
        self.result = None


class Bess:
    def __init__(self, id, bus, e_cap_kwh, p_charge_max_kw, p_discharge_max_kw,
                 eta_charge, eta_discharge, soc_init_frac, soc_min_frac, soc_max_frac, cyclic_soc):
        self.id, self.bus, self.e_cap_kwh = id, bus, e_cap_kwh
        self.p_charge_max_kw, self.p_discharge_max_kw = p_charge_max_kw, p_discharge_max_kw
        self.eta_charge, self.eta_discharge = eta_charge, eta_discharge
        self.soc_init_frac, self.soc_min_frac, self.soc_max_frac = soc_init_frac, soc_min_frac, soc_max_frac
        self.cyclic_soc = cyclic_soc
        self.result = None


class Pv:
    def __init__(self, id, bus, p_max_kw, curtailable, power_factor, avail_kw):
        self.id, self.bus, self.p_max_kw = id, bus, p_max_kw
        self.curtailable, self.power_factor = curtailable, power_factor
        self.avail_kw = avail_kw
        self.result = None


class Summary:
    def __init__(self, status, objective_cost, energy_cost, energy_import_kwh, energy_export_kwh,
                 pv_generated_kwh, pv_curtailed_kwh, losses_kwh, v_min_pu, v_max_pu):
        self.status, self.objective_cost, self.energy_cost = status, objective_cost, energy_cost
        self.energy_import_kwh, self.energy_export_kwh = energy_import_kwh, energy_export_kwh
        self.pv_generated_kwh, self.pv_curtailed_kwh = pv_generated_kwh, pv_curtailed_kwh
        self.losses_kwh = losses_kwh
        self.v_min_pu, self.v_max_pu = v_min_pu, v_max_pu

    def as_dict(self): return dict(self.__dict__)


class Case:
    def __init__(self, name, base, buses, branches, grid, bess, pv, evse,
                 timestamps, dt_h, price, objective):
        self.name, self.base = name, base
        self.buses, self.branches, self.grid = buses, branches, grid
        self.bess, self.pv, self.evse = bess, pv, evse
        self.timestamps, self.dt_h, self.price = timestamps, dt_h, price
        self.objective = objective
        self.parent_branch, self.children = {}, {}    # topologia radial (derivada)
        self.summary = None

    @property
    def periods(self): return range(len(self.timestamps))
    @property
    def n_periods(self): return len(self.timestamps)
    @property
    def root(self): return self.grid.bus
    @property
    def index(self): return pd.DatetimeIndex(self.timestamps)
