# opf-teacher

Optimal operation of batteries (BESS) and DERs in **distribution networks** for
cost reduction, via **OPF** (optimal power flow).

The model uses the **branch flow model (DistFlow, Baran-Wu)** for radial networks
with a **second-order cone relaxation (SOCP)** — the exact rotated cone:

```
P_j² + Q_j²  ≤  v_i·ℓ_j                                    (exact rotated cone)
v_j = v_i − 2(r_j P_j + x_j Q_j) + (r_j² + x_j²) ℓ_j       (voltage drop)
soc_t = soc_{t−1} + (η_c·pch − pdis/η_d)·Δt                (battery dynamics)
min  Σ_t  price_t · (import_t − r_feedin·export_t) · Δt    (energy cost)
```

The full formulation (indices, parameters, variables, objective and all
constraints) is in [docs/formulation.md](docs/formulation.md).

## Layout

```
opf/
├── components.py  domain classes (Base, Bus, Branch, Grid, Bess, Pv, Case):
│                  physical parameters + a .result filled after the solve
├── data.py        load_case(dir) -> Case   (reads JSON + CSV)
├── model.py       build_model(case) -> Pyomo model (DistFlow SOCP, converts to p.u.)
├── pv_optimal.py  PV inverter control via optimization (optimal / fixed_pf)
├── pv_droop.py    autonomous PV control: Volt-VAr / Volt-Watt curves (SOS2)
└── results.py     attach_results(model, case) -> fills .result and .summary
teacher.py         BessOpt facade  ->  BessOpt(dir).build().solve()
run.py             solve + save a plot panel to figures/
examples/
└── case5/         a 5-bus radial feeder, 1 BESS, 1 PV
```

Each component holds its **parameters** (physical units) and, after the solve,
its **result** in `.result` (series indexed by timestamp). The per-unit
conversion is centralized in `Base` and happens inside the model.

### Anatomy of an example (`examples/case5/`)

| File | Role |
|---|---|
| `config.json`  | system/grid: p.u. base (kVA/kV), grid/substation, voltage limits, objective |
| `devices.json` | controllable DERs: `bess`, `pv`, `evse` (which exist + static parameters) |
| `bus.csv`      | buses: `bus_id, type, name` |
| `branch.csv`   | radial branches: `from_bus, to_bus, r_ohm, x_ohm, s_max_kva` |
| `demand.csv`   | load (wide): `timestamp, P2, Q2, …, Pn, Qn` in kW / kVAr |
| `pv.csv`       | PV availability: `timestamp, PV<bus>` in kW |
| `price.csv`    | tariff: `timestamp, price_per_kwh` |

Unit convention in the files: **kW / kVAr / kWh / Ω / kV**. The loader converts
everything to per-unit on the system base (`config.base`); the time axis (periods
and `Δt`) is inferred from the series `timestamp`s.

## Usage

```python
from teacher import BessOpt

case = BessOpt("examples/case5").build().solve(solver="gurobi_direct")

print(case.summary)                    # cost, energy, losses, min/max voltage
print(case.bess[0].result.soc_kwh)     # battery SoC (kWh) per timestamp
print(case.bess[0].result.p_net_kw)    # charge (+) / discharge (-) (kW)
print(case.grid.result.import_kw)      # grid import (kW)
print(case.buses[4].result.v_pu)       # voltage (p.u.) at bus 4
```

`solve()` returns the `Case` with every component populated: `case.grid`,
`case.bess[i]`, `case.pv[i]`, `case.buses[b]`, `case.branches[i]` — each with a
`.result` in kW/kVAr/kV/kWh.

### Visualization

`run.py` solves an example and saves a panel (price vs import, battery vs SoC,
PV vs load, voltages) to `figures/`:

```bash
python run.py                       # examples/case5
python run.py examples/case5 --show # open the matplotlib window
```

## PV inverter control

Each PV chooses how its inverter dispatches P and Q via the `control` field in
`devices.json`:

| `control` | Q | P |
|---|---|---|
| `optimal`       | free within the capability disk (optimizer's choice) | curtailable |
| `fixed_pf`      | `Q = tan(φ)·P` | curtailable |
| `volt-var`      | Volt-VAr droop curve `Q(V)` | at availability |
| `volt-watt`     | 0 | Volt-Watt droop curve `P ≤ f(V)` |
| `volt-var-watt` | Volt-VAr `Q(V)` | Volt-Watt `P(V)` |

The inverter rating `P² + Q² ≤ S²` applies in every mode. The droop curves
(`pv_droop.py`) are the standard IEEE-1547 local controls; `optimal`
(`pv_optimal.py`) lets the OPF pick the best P/Q and generally dominates them.

## Solver

The problem is a **MISOCP**: linear objective, a conic constraint (SOCP), and one
**SOS1** constraint per period (no simultaneous import/export). The solver must
support quadratic cones **and** SOS / branch-and-bound — pure LP/MILP (e.g. HiGHS)
and pure continuous NLP (e.g. IPOPT) do **not** work. Options:

- **Gurobi** (`gurobi_direct`) — the default. Handles SOCP + SOS1 natively.
- **SCIP** (`scip` / PySCIPOpt) — open-source, MINLP/MISOCP.

### Gurobi license (academic, free)

The `pip`-installed `gurobipy` ships with a **restricted license** that only
solves small models (the full 24h `case5` exceeds it because of the conic
constraints). To unlock it:

1. Sign in at <https://portal.gurobi.com> with an academic email.
2. Request an **Academic WLS** (Web License Service) license — this is the one
   that works with pip's `gurobipy`, without installing the full Gurobi.
3. Download `gurobi.lic` (it holds `WLSACCESSID`, `WLSSECRET`, `LICENSEID`) and
   put it at `C:\Users\<you>\gurobi.lic`, or export the matching env vars.

Check with:

```python
import gurobipy as gp
gp.Env().start()   # no error => license is active
```
