# opf-teacher

Optimal operation of batteries and distributed energy resources in radial
distribution networks. The network is read from an OpenDSS `Master.dss` file;
loads, prices, and PV availability are provided as CSV time series.

The optimization model uses DistFlow with an SOCP relaxation. See the complete
formulation in [English](docs/formulation.md) or
[Portuguese](docs/formulation.pt-BR.md).

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Gurobi is the default solver. To share a WLS license across projects, place
`gurobi.lic` at:

```text
C:\Users\<username>\gurobi.lic
```

## Running the example

Paths and runtime options are defined at the top of `main.py`:

```python
CASE_PATH = PROJECT_ROOT / "examples" / "case5"
OUTPUT_PATH = PROJECT_ROOT / "figures" / "case5_dispatch.png"
SOLVER = "gurobi_direct"
SOCP_GAP_TOLERANCE = 1e-5
SHOW_PLOT = False
```

Run:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Input files

The `case5` directory is the reference example:

```text
examples/case5/
├── config.json
├── devices.json
├── demand.csv
├── price.csv
├── pv.csv
└── dss/
    └── Master.dss
```

| File | Contents |
|---|---|
| `config.json` | system bases, slack bus, voltage limits, and grid limits |
| `devices.json` | BESS and PV parameters |
| `dss/Master.dss` | buses, lines, transformers, impedances, and ratings |
| `demand.csv` | active and reactive demand by bus |
| `price.csv` | energy price by period |
| `pv.csv` | available PV power |

Input units are kW, kVAr, kWh, ohm, and kV. Conversion to per unit is performed
when the Pyomo model is built.

The network path and slack bus are configured in `config.json`:

```json
"network": {
  "master": "dss/Master.dss",
  "slack_bus": "bus_001"
}
```

OpenDSS bus names are also used in `devices.json` and demand columns, such as
`Pbus_004` and `Qbus_004`. Names without a numeric suffix can be mapped
explicitly:

```json
"bus_ids": {
  "source": 1,
  "load": 2
}
```

## OpenDSS support

Supported network elements:

- radial topology;
- three-phase lines using `R1`, `X1`, `Length`, and `NormAmps`;
- three-phase, two-winding transformers;
- transformer `kV`, `kVA`, `%R`, `XHL`, connection, and fixed tap;
- multiple voltage levels.

Not yet supported:

- meshed networks;
- single-phase transformer banks;
- transformers with three or more windings;
- automatic `RegControl` actions;
- unbalanced three-phase OPF.

Line `Rmatrix`, `Xmatrix`, and phase data are retained in the network objects
for the future unbalanced formulation.

## Python API

```python
from teacher import BessOpt

case = BessOpt("examples/case5").build().solve(
    solver="gurobi_direct",
    socp_gap_tolerance=1e-5,
)

print(case.summary.as_dict())
print(case.bess[0].result.soc_kwh)
print(case.bess[0].result.p_net_kw)
print(case.bess[0].result.q_kvar)
print(case.bess[0].result.inverter_loss_kw)
print(case.pv[0].result.inverter_loss_kw)
print(case.pv[0].result.p_net_kw)
print(case.pv[0].result.grid_consumption_kw)
print(case.grid.result.import_kw)
print(case.buses[4].result.v_pu)
```

The BESS sign convention is `p_net_kw > 0` for charging and `p_net_kw < 0`
for discharging. Reactive power uses `q_kvar > 0` for injection and
`q_kvar < 0` for absorption.

## BESS inverter

Reactive-power control is configured per BESS in `devices.json`:

```json
{
  "e_cap_kwh": 100.0,
  "p_charge_max_kw": 40.0,
  "p_discharge_max_kw": 40.0,
  "s_max_kva": 50.0,
  "reactive_control": true,
  "q_loss_rated_kw": 0.5
}
```

When enabled, the OPF selects BESS reactive power subject to the inverter
rating. If `reactive_control` is omitted or `false`, reactive power is fixed at
zero. If `s_max_kva` is omitted, it defaults to the largest active-power limit.
`q_loss_rated_kw` is the incremental active-power loss at
`abs(q_kvar) == s_max_kva`; the loss scales quadratically with Q. For the BESS,
this loss is subtracted from stored energy. For PV, it consumes part of the
available solar power. Default: zero.

Per-device loss series are exposed as `result.inverter_loss_kw`; their total
energy is reported in `case.summary.inverter_losses_kwh`.

## Relaxation gap

After each solve, the code evaluates

```text
g = v*l - P² - Q²
```

The main fields in `case.summary` are:

- `socp_gap_max_normalized`;
- `socp_gap_max_relative_flow`;
- `socp_current_error_max_a`;
- `socp_loss_error_max_w`;
- `socp_gap_tolerance`;
- `socp_tightness_margin`;
- `socp_tightness`;
- `socp_relaxation_tight`.

The tightness margin is `tolerance - maximum normalized gap`. A positive value
passes the configured criterion. `socp_tightness` is `tight`, `acceptable`, or
`not_tight`. Per-branch time series include the absolute and normalized gaps,
gap relative to branch flow, current error in amperes, and equivalent loss
error in watts.

This is a numerical and physical tightness check, not a statistical
probability. `socp_confidence` and `socp_confidence_margin` remain as aliases.

## PV control

The `control` field accepts:

| Value | Behavior |
|---|---|
| `optimal` | P and Q selected by the OPF |
| `fixed_pf` | fixed power factor |
| `volt-var` | local Volt-VAr curve |
| `volt-watt` | local Volt-Watt curve |
| `volt-var-watt` | combined Volt-VAr and Volt-Watt curves |

All modes enforce the inverter apparent-power rating.
PV devices also accept `q_loss_rated_kw`. With a nonzero value, reactive-power
losses reduce the active power available at the AC terminal.

Night-time reactive support is optional:

```json
{
  "s_max_kva": 300.0,
  "q_loss_rated_kw": 3.0,
  "night_var": true
}
```

`night_var` defaults to `false` and requires a positive `q_loss_rated_kw` when
enabled. At night, the PV supplies Q and imports its inverter loss from the
grid. The result fields are `grid_consumption_kw` and `p_net_kw`. Supported
controls are `optimal`, `volt-var`, and `volt-var-watt`.

## Repository layout

```text
opf/
├── components.py   domain and result objects
├── data.py         case loader
├── opendss.py      OpenDSS interface
├── model.py        Pyomo formulation
├── pv_droop.py     Volt-VAr and Volt-Watt controls
├── pv_optimal.py   optimal and fixed_pf controls
└── results.py      result conversion and gap analysis
teacher.py          BessOpt interface
main.py             configured example runner
```
