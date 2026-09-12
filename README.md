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

For use as a module from a separate training repository:

```powershell
python -m pip install -e path\to\opf-teacher
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
ACTIVE_POWER_ONLY = True
```

Run:

```powershell
.\.venv\Scripts\python.exe main.py
```

With `ACTIVE_POWER_ONLY = True`, `main.py` also writes the devices, buses,
branches, and summary CSVs under `results/`.

The demand-unbalanced three-phase example uses the phase-native AC-IVR solver:

```powershell
.\.venv\Scripts\python.exe scripts\run_three_phase_opf.py
```

It writes `case5_unbalanced_{devices,buses,branches,summary}.csv` under
`results/`. This solver uses SciPy SLSQP and reports a physically feasible local
solution, without a global-optimality certificate.

## Active-power CSV export

An active-power-only LinDistFlow variant is available for producing a simple
teacher trajectory. It optimizes grid exchange, BESS charge/discharge, BESS
state of charge, PV generation, branch active flows, and bus voltage
magnitudes. Reactive power and network losses are omitted from this linear
formulation.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_active_power_opf.py
```

Outputs:

```text
results/case5_active_power_opf_devices.csv
results/case5_active_power_opf_buses.csv
results/case5_active_power_opf_branches.csv
results/case5_active_power_opf_summary.csv
```

Each CSV uses one row per timestep and one column per measured quantity. The
device CSV contains BESS powers and state of charge, plus PV availability,
generation, and curtailment. The bus CSV contains loads, voltages, load
currents, prices, and grid exchange. The branch CSV contains active and
reactive flows, branch currents, and losses.

Device net active power follows `p_net_kw = power received from the grid -
power injected into the grid`. Therefore, positive values represent net
consumption and negative values represent net injection.

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

Every public loader accepts either the case directory or the configuration
file itself. This allows an external repository to own one case and pass the
same source to the teacher and OpenDSS environment:

```python
case_source = "scenarios/network_01/config.json"  # directory also accepted
teacher = Teacher(case_source)  # formulation may be read from config.json
```

Paths are resolved relative to the configuration file. The optional `files`
section can override the default data filenames:

```json
"schema_version": 1,
"files": {
  "demand": "demand.csv",
  "prices": "price.csv",
  "devices": "devices.json"
}
```

Without this section, the conventional filenames are used. The examples under
`examples/` follow the same contract and remain self-contained.

Case files use `"schema_version": 1`. Runtime observations use the independent
`"observation_schema_version": 1` contract.

Input units are kW, kVAr, kWh, ohm, and kV. Conversion to per unit is performed
when the Pyomo model is built.

The reference `case5` is the single-phase equivalent of a balanced three-phase
system. Its kW and kVAr values are system totals, while line currents follow the
single-phase equivalent used by its OpenDSS model.

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
- single-phase equivalent or balanced three-phase lines using `R1`, `X1`,
  `Length`, and `NormAmps`;
- phase-native radial lines using full `Rmatrix` and `Xmatrix`, including mutual
  coupling;
- three-phase, two-winding transformers;
- transformer `kV`, `kVA`, `%R`, `XHL`, connection, and fixed tap;
- multiple voltage levels.

Not yet supported:

- meshed networks;
- single-phase transformer banks;
- transformers with three or more windings;
- automatic `RegControl` actions;
- transformers, delta devices, and explicit neutrals in the phase-native model.

The initial phase-native formulation supports grounded-wye-equivalent cases
whose neutral has already been Kron-reduced into the line matrices. Imbalance
is specified explicitly in phase demand columns.

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

The formulation can also be selected explicitly without changing the legacy
`active_power_only` API:

```python
from teacher import Teacher

active = Teacher("examples/case5", formulation="single_phase_active")
full = Teacher("examples/case5", formulation="single_phase_socp")
unbalanced = Teacher(
    "examples/case5_unbalanced",
    formulation="three_phase_ivr",
).solve()
```

For phase-native observations, use the explicit teacher boundary:

```python
from teacher import ThreePhaseTeacher
from opf.three_phase import BusState, BessState, PvState

teacher = ThreePhaseTeacher("examples/case5_unbalanced")
solved = teacher.observe(
    buses=bus_states,
    bess=bess_states,
    pv=pv_states,
).solve()
x, y = solved.state_action()
```

The named Gymnasium/OpenDSS observation can be consumed without a conversion
layer:

```python
_, reset_info = env.reset()
solved = ThreePhaseTeacher(case_source).observe_dict(
    reset_info["observation"]
).solve()
_, action = solved.state_action()
next_state, reward, terminated, truncated, info = env.step(action)
```

Three-phase bus, BESS, PV, and action values are dictionaries indexed by
`a`, `b`, and `c`. The temporal-window behavior is the same as for the
single-phase teacher.

The existing models are single-phase equivalents for balanced systems and live
under `opf/single_phase/`. Phase-native buses, branches, device connections,
states, actions, and results are available under `opf/three_phase/`; see the
[three-phase contract](docs/three-phase-contract.md). The phase-native loader
requires `P<bus>_<phase>` and `Q<bus>_<phase>` demand columns and the AC-IVR
solver retains the full line impedance matrices.

The BESS sign convention is `p_net_kw > 0` for charging and `p_net_kw < 0`
for discharging. Reactive power uses `q_kvar > 0` for injection and
`q_kvar < 0` for absorption.

## Teacher state-action pairs

Run `python example.py` to compare two independent five-bus scenarios:

- `examples/case5_central`: BESS `b1` at bus 4 and `b2` at bus 2;
  PV `pv1` at bus 5 and `pv2` at bus 3. Central X includes every bus and
  device; Y contains all four devices. The second PV profile is half the first.
- `examples/case5_local`: only BESS `b1` at bus 4, with no PV.
  Local X contains bus 4 and battery state; Y contains only battery commands.

Both use the reference topology, demand and prices. The example uses HiGHS
with active power only and illustrative pre-action voltages of 1 pu.
The example advances through `STEPS` horizons and builds separate temporal
windows of size `WINDOW` for central and local control. Each label is the first
action of that horizon. Observations represent an illustrative idle history:
SoC stays constant and previous commands are zero. Teacher actions are labels
only, not applied commands. In an actual rollout, replace these observations
with environment measurements. Each teacher solves its full network even when
the student's observation is local.

`Teacher` uses the existing `Case`, `Bus`, `Bess`, `Pv`, and `Grid` objects.
`BessOpt` remains available with the same build/solve usage. Components now
have `.state` for pre-action observations; BESS and PV also have `.action`
for the first optimal command. `.result` still contains the full predicted
trajectory. Grid exchange is a result, not a controllable device action.

The integration/training repository supplies observations from its environment:

```python
from teacher import Teacher
from opf import BusState, BessState, PvState


def label_horizon(case, bus_states, bess_states, pv_states):
    teacher = Teacher(case, active_power_only=True)
    solved = teacher.observe(
        buses=bus_states,       # {integer bus ID: BusState(...)}
        bess=bess_states,       # {device ID: BessState(...)}
        pv=pv_states,           # {device ID: PvState(...)}
    ).solve()                  # HiGHS for active mode; Gurobi for full mode

    central_x, central_y = solved.state_action()
    local_pairs = {
        "bess": {d.id: d.state_action(solved.buses[d.bus]) for d in solved.bess},
        "pv": {d.id: d.state_action(solved.buses[d.bus]) for d in solved.pv},
    }
    return (central_x, central_y), local_pairs, solved.summary
```

Observation constructors (scalar values for one instant):

- `BusState(v_before_pu, p_load_kw, q_load_kvar)`;
- `BessState(soc_before_frac, previous_p_kw, previous_q_kvar)`;
- `PvState(available_kw, previous_p_kw, previous_q_kvar)`.

`observe()` requires every bus and device, including empty dictionaries when
a device type is absent. Bus IDs are the integer keys of `case.buses`; the
loader exposes `case.bus_name_to_id` for converting OpenDSS names. Previous
BESS P is positive for charging; previous PV P uses net injection at its AC
terminal; previous device Q is positive for injection.

All observations must refer to the **first timestamp of the supplied Case**.
`observe()` updates the first load/PV profile values and initial BESS SoC,
invalidates previous solutions, and leaves later profile values as forecasts.
It does not advance time or calculate a power flow. The caller supplies a new
Case with the next horizon when advancing the episode. Pre-action voltage
must be measured or calculated by the environment with the prior applied
commands; it is never inferred from optimized voltage or fixed in the OPF.

The optional BESS parameter `soc_terminal_frac` specifies an end-of-horizon
target even when `cyclic_soc` is false. Without an explicit target, ordinary
solves retain the existing cyclic behavior. When observing a new SoC in a
cyclic case, `observe()` preserves the original initial SoC as the terminal
target. Set a new target explicitly when changing the terminal policy.

`state_action()` returns named dictionaries, not normalized vectors:

- Central X: timestamp, dt, all buses, BESS/PV states and grid tariffs.
- Central Y: BESS and PV action dictionaries keyed by device ID.
- Local X: only the connected bus state and the device's own state.
- Local Y: `p_net_kw`/`q_injection_kvar` for BESS, or
  `generation_kw`/`q_injection_kvar` for PV. PV generation is positive;
  nighttime grid consumption is calculated separately by the environment.

Local pairs intentionally exclude tariffs, forecasts and other devices.
The training repository may add information available to its controller,
including time, terminal targets and forecast windows. These extra features
are especially relevant to reproduce a teacher that sees the full horizon.
It also owns vector ordering, normalization and dataset storage. Local
labels remain actions from a global optimization, not a separately optimized
local policy.

Only optimal/locally optimal solver terminations produce actions. This first
interface does not certify AC feasibility, SOCP tightness, or replay equality;
use the returned summary and an OpenDSS replay to accept training labels.

## Temporal pairs

`TemporalTeacher` builds a sliding window from successive state-action pairs.
It accepts either central or local dictionaries and keeps the current action
as the target. Use one instance per episode/controller stream.

```python
from teacher import TemporalTeacher

temporal = TemporalTeacher(window=3)

# At each observed step, after solving the corresponding horizon:
x, y = case.state_action()  # Or battery.state_action(case.buses[battery.bus])
pair = temporal.append(x, y)
if pair is not None:
    x_history, y_current = pair
    # x_history = [x_t_minus_2, x_t_minus_1, x_t]

# Before starting another episode:
temporal.reset()
```

The first `window - 1` calls return `None`. Returned pairs are independent
copies. The caller supplies observations in chronological order at the chosen
sampling interval; the class does not check timestamps or advance the environment.
Do not fill the window with the OPF's predicted future trajectory.

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
├── single_phase/   existing balanced-equivalent formulations and exports
├── three_phase/    unbalanced AC-IVR formulation, loader, and exports
├── components.py   shared domain and result objects
├── data.py         shared case loader
├── formulations.py formulation names and compatibility rules
├── opendss.py      shared OpenDSS interface
├── model.py        compatibility import for single_phase/distflow_socp.py
└── active_model.py compatibility import for single_phase/active_model.py
teacher.py          BessOpt interface
main.py             configured example runner
```
