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
.\.venv\Scripts\python.exe -m pip install -e ..\opendss_environment
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The environment integration requires the `opendss_environment` checkout with
the version 0.5 package API. The command above assumes the two repositories
are sibling folders. Install that checkout before the teacher requirements
or the optional `environment` extra.

For use as a module from a separate training repository:

```powershell
python -m pip install -e path\to\opf_teacher
```

The Python package is `opf_teacher`; the distribution name remains `opf-teacher`.
Use explicit package imports:

```python
from opf_teacher.teacher import Teacher
from opf_teacher.opf.single_phase import load_case
```

`from opf_teacher import Teacher` is also available. Version 0.3 moves the old
top-level `teacher`, `normalization`, and `opf` modules under `opf_teacher`.
Update external imports and reinstall the editable project after this migration.

For the active-power training scripts, install the environment and HiGHS extras
from this repository's root:

```powershell
python -m pip install -e ".[highs,environment]"
```

`requirements.txt` also includes the environment. Install this project with
`python -m pip install -e .` to make its package available from other folders.
The environment is developed in its
[own repository](https://github.com/TeradaZenichi/powerflow-opendss-environment)
and installed from the sibling checkout for this package migration. Scenario inputs
come from `examples/`, and current outputs go to `results/`. Local `.old/`
archives are ignored by Git and are not used by the active scripts.

The SOCP formulation requires Gurobi or an equivalent compatible solver.

## Running

`main.py` solves the battery OPF over the full horizon for `examples/case5_local` using the
active-power LinDistFlow formulation. It writes device, bus, branch and summary CSVs
as `results/battery_opf_{devices,buses,branches,summary}.csv`.

```powershell
.\.venv\Scripts\python.exe main.py
```

The script only optimizes and exports trajectories. It does not run an external
environment or generate training windows.

Change run parameters in the block at the top of each script:

| Parameter | Meaning |
|---|---|
| `CASE_PATH` | Scenario directory under `examples/`. |
| `FORMULATION` | `single_phase_active` or `single_phase_socp`; `train.py` also accepts `three_phase_ivr`. |
| `OBSERVATION_MODE`, `DEVICE_ID` | `train.py`: central observations or a local view of the selected BESS. |
| `RESULTS_PATH`, `OUTPUT_NAME` | Output directory and CSV prefix in `main.py`. |
| `CSV_PATH` | Output CSV path in each `train` script. |
| `WINDOW` | Positive number of observed states in each X. A longer window than the episode produces zero pairs. |

The scenario's CSV timestamps define the number of periods and their duration.
Change demand and prices in `demand.csv` and `price.csv`, battery parameters in
`devices.json`, and network limits in `config.json`. The scripts use all periods.
The training scripts use `base.s_base_kva` and record `base.v_base_kv` from
`config.json`; change these fields to select the scenario bases.

The demand-unbalanced three-phase example uses the three-phase AC-IVR solver:

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
`"observation_schema_version": 2` contract, with `units: "pu"` and explicit bases.

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
- three-phase radial lines using full `Rmatrix` and `Xmatrix`, including mutual
  coupling;
- three-phase, two-winding transformers;
- transformer `kV`, `kVA`, `%R`, `XHL`, connection, and fixed tap;
- multiple voltage levels.

Not yet supported:

- meshed networks;
- single-phase transformer banks;
- transformers with three or more windings;
- automatic `RegControl` actions;
- ungrounded `wye-delta`/`delta-delta` transformer secondaries and explicit
  neutrals in the three-phase model.

The three-phase formulation supports grounded-wye loads and `wye` or `delta`
controllable devices. For delta devices, action keys `a`, `b`, and `c` represent
the `a-c`, `b-a`, and `c-b` legs. Explicit neutrals remain Kron-reduced into the
line matrices. Imbalance is specified explicitly in phase demand columns.

The three-phase AC-IVR solver supports multiple BESS, aggregate or independent
per-phase dispatch, BESS reactive control and inverter losses, optimal/fixed-PF
PV operation, Volt-VAr, Volt-Watt and combined curves. Three-phase two-winding
`wye-wye` and `delta-wye` transformers use their fixed tap and phase shift while
preserving each bus voltage base.

## Python API

```python
from opf_teacher.teacher import BessOpt

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

Select a formulation using one of its three official names:

```python
from opf_teacher.teacher import Teacher

active = Teacher("examples/case5", formulation="single_phase_active")
full = Teacher("examples/case5", formulation="single_phase_socp")
unbalanced = Teacher(
    "examples/case5_unbalanced",
    formulation="three_phase_ivr",
).solve()
```

For three-phase observations, use `ThreePhaseTeacher`:

```python
from opf_teacher.teacher import ThreePhaseTeacher
from opf_teacher.opf.three_phase import BusState, BessState, PvState

teacher = ThreePhaseTeacher("examples/case5_unbalanced")
solved = teacher.observe(
    buses=bus_states,
    bess=bess_states,
    pv=pv_states,
).solve()
x, y = teacher.state_action()
```

The named Gymnasium/OpenDSS observation can be consumed without a conversion
layer:

```python
observation, _ = env.reset()
teacher = ThreePhaseTeacher(case_source)
teacher.observe_dict(observation)
teacher.solve()
_, action = teacher.state_action()
next_state, reward, terminated, truncated, info = env.step(action)
```

`train.py` supports central and device-local observations for the balanced
single-phase equivalent and the unbalanced three-phase model.

Three-phase bus, BESS, PV, and action values are dictionaries indexed by
`a`, `b`, and `c`. The temporal-window behavior is the same as for the
single-phase teacher.

The existing models are single-phase equivalents for balanced systems and live
under `opf_teacher/opf/single_phase/`. Three-phase buses, branches, device connections,
states, actions, and results are available under `opf_teacher/opf/three_phase/`; see the
[three-phase contract](docs/three-phase-contract.md). The three-phase loader
requires `P<bus>_<phase>` and `Q<bus>_<phase>` demand columns and the AC-IVR
solver retains the full line impedance matrices.

The BESS sign convention is `p_net_kw > 0` for charging and `p_net_kw < 0`
for discharging. Reactive power uses `q_kvar > 0` for injection and
`q_kvar < 0` for absorption.

## Teacher state-action pairs

`train.py` generates X/Y pairs from an OpenDSS episode for any positive history
window. Change `WINDOW` in the settings block at the top of the script:

```powershell
.\.venv\Scripts\python.exe train.py
```

Install the `environment` extra in the same Python environment first.
The default settings use `case5_local` (one battery, no PV), solve the
active-power teacher at every step and execute its first action in OpenDSS.
Observed states include the resulting SoC and previously applied device powers.
Set `CASE_PATH` and `FORMULATION` together when selecting another scenario;
the script also accepts SOCP and three-phase cases.

`reset()` returns the initial named observation in pu. Each `step(action)`
accepts the global action in pu and returns the next named observation in pu,
ready for `teacher.observe_dict(observation)` at the next timestamp. `observe()`
returns the same current snapshot without advancing time. At episode end,
`step()` returns `None` for the next observation and sets `terminated=True`.
`info["observation_before"]` records the observation used for the executed
action; the other step measurements describe the executed snapshot.

The environment and teacher use the same standalone `pu-normalizer` package.
Their public observations, X/Y pairs and step power measurements use pu.
Conversions to physical powers happen internally at the OPF/OpenDSS boundary.
The training script only selects the local/central view and encodes time before
adding X/Y to the history. Pass the teacher or NN action directly to `env.step()`.
`env.normalizer.denormalize(info)` provides physical powers for reports when needed.

An integration repository can keep the shared scenario under `data/case5/`
and pass that directory (or its configuration file) to both libraries. Neither
library depends on the directory name `examples`. The environment's
`EnvironmentConfig(case_path)` uses all case periods by default; an explicit
`episode_steps` still selects an episode length.

The environment implementation lives in `opendss_environment/microgrid_env.py`
inside its own repository, alongside its `main_env.py` example runner:

```python
from opendss_environment import EnvironmentConfig, MicrogridEnv
```

Its `opendss_environment/` package contains the environment and supporting modules.
Use version 0.5 or later of the `opendss-environment` distribution.

The environment accepts `Action`, `TotalAction` and `PhaseAction` objects in pu,
as well as the named dictionaries from `Teacher.state_action()`. Every controlled
device must have a command. Unequal phase commands require per-phase dispatch;
they are never silently aggregated. Execution failures leave the public time,
SoC and histories at the last successful step and require `reset()` before
continuing, since the external DSS context may have changed.

Declare `network.representation` in new case configurations:

- `single_phase`: a physical single-phase network, with a line-to-neutral base;
- `single_phase_equivalent`: the existing single-phase equivalent, whose
  powers represent system totals;
- `three_phase`: the physical three-phase representation, with phase values.

Actual single-phase examples require consistent DSS bus voltage bases; see
`tests/fixtures/single_phase_case`. The single-phase teacher uses the configured
voltage base and derives its line rating as `V_base_kV * NormAmps`. Existing
equivalent cases retain their configured base. Three-phase balanced models
retain their line-to-line conversion. Single-phase transformer banks remain
outside the teacher's supported network contract.

Direct observation exchange requires matching network representations.
When a single-phase equivalent teacher controls a three-phase environment,
the integration repository owns the observation and action adaptation.
The environment remains independent of OPF formulations and learning code.

The outputs are `results/train.csv` and `results/train.json`. The JSON records
the scenario, formulation, observation mode, selected BESS/bus, window, phase order, number of pairs, bases and time
encoding. Phase order stays in this metadata so the columns under X remain
numeric. You can also import the script and call `main(window=5)`.

The formulation and observation scope are independent settings:

```python
FORMULATION = "single_phase_active"  # Or single_phase_socp / three_phase_ivr.
OBSERVATION_MODE = "local"  # Or central.
DEVICE_ID = "b1"  # Used only in local mode.
WINDOW = 3
```

| Observation mode | X in each history position | Y |
|---|---|---|
| `central` | All buses, BESS/PV states, tariffs, interval length and encoded time | All BESS/PV actions |
| `local` | The selected BESS's bus, its own SoC and previous powers, interval length and encoded time | Only that BESS's action |

`DEVICE_ID` refers to a BESS in `devices.json`. The local view includes the
connected bus's voltage and P/Q load, plus phase angles for three-phase cases.
It excludes other buses/devices and tariffs. `PairSelector` in `teacher.py`
builds this view from the central pair, using the same global OPF label.
The timestamp and phase order remain metadata; the training script encodes
the time before adding the selected state to the history.

Both modes solve the complete OPF and execute every device's action
in OpenDSS, including during warm-up. The selection changes only the exported
pairs. Single-phase pairs use scalars, three-phase powers/voltages keep their
phase keys, and each battery keeps one scalar SoC.

Central CSV targets use names such as `y.bess.b1.p_net_pu`. Local CSV targets
use `y.p_net_pu`; local inputs use `x.0.bus.*` and `x.0.device.*` instead of the
central `x.0.buses.<id>.*` and `x.0.bess.<id>.*` paths.

An episode with N periods produces `max(0, N - WINDOW + 1)` pairs:

| Window | Pairs for the 24-period case |
|---|---|
| 1 | 24 |
| 3 | 22 |
| 5 | 20 |
| 24 | 1 |
| 25 | 0 |

The first `WINDOW - 1` steps collect history while still executing the teacher
actions. A window longer than the episode writes a CSV with only its header
and records zero pairs in the JSON.

The previous scripts remain available:

| Script | Window | Pairs for the 24-period case | Output |
|---|---|---|---|
| `train-w1.py` | 1 | 24 | `results/train-w1.csv` |
| `train-w3.py` | 3 | 22 | `results/train-w3.csv` |

Each CSV row contains `timestamp`, chronological state columns `x.0.` (and
`x.1.` and `x.2.` for window three), and the current teacher action under `y.`. Actions are
executed during warm-up as well. These scripts generate labels, not fitted
learning models. Forecast trajectories do not populate the observed history.
The timestamp string is row metadata. Each state in X contains numeric
`time.hour_sin` and `time.hour_cos` features for its own observation time.

The training CSVs store powers in pu in both X and Y. Each CSV has a JSON with
the same name containing the scenario, window, normalization bases and time encoding. Device
commands are converted internally to kW/kvar for OpenDSS. The result CSVs from `main.py`
continue to use the physical units shown in their column names.

`Teacher` uses the existing `Case`, `Bus`, `Bess`, `Pv`, and `Grid` objects.
`BessOpt` remains available with the same build/solve usage. Components now
have `.state` for pre-action observations; BESS and PV also have `.action`
for the first optimal command. `.result` still contains the full predicted
trajectory. Grid exchange is a result, not a controllable device action.

The integration/training repository supplies observations from its environment:

```python
from opf_teacher.teacher import PairSelector, Teacher


def label_horizon(case, observation, device_id):
    teacher = Teacher(case, formulation="single_phase_active")
    teacher.observe_dict(observation)
    teacher.build()
    solved = teacher.solve()

    central_x, central_y = teacher.state_action()
    selector = PairSelector(case, "local", device_id)
    local_pair = selector.select(central_x, central_y)
    return (central_x, central_y), local_pair, solved.summary
```

For direct access to the model, `observe()` also accepts state classes in
physical units (scalar values for one instant in the single-phase case):

- `BusState(v_before_pu, p_load_kw, q_load_kvar)`;
- `BessState(soc_before_frac, previous_p_kw, previous_q_kvar)`;
- `PvState(available_kw, previous_p_kw, previous_q_kvar)`.

`soc_before_frac` is one scalar between 0 and 1 per battery in both formulations.
Three-phase batteries share this energy state across their phase powers.

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

`Teacher.state_action()` returns named dictionaries in pu:

- Central X: timestamp, dt, all buses, BESS/PV states and grid tariffs.
- Central Y: BESS and PV action dictionaries keyed by device ID.
- Local X: only the connected bus state and the device's own state.
- Local Y selected by `PairSelector`: `p_net_pu`/`q_injection_pu` for BESS.
  Central PV actions use `generation_pu`/`q_injection_pu`. PV generation is positive;
  nighttime grid consumption is calculated separately by the environment.

Local pairs intentionally exclude tariffs, forecasts and other devices.
The training repository may add information available to its controller,
including time, terminal targets and forecast windows. These extra features
are especially relevant to reproduce a teacher that sees the full horizon.
It also owns vector ordering and any statistical preprocessing used by its
learning model. Both central and local pairs already use pu. Local labels
remain actions from a global optimization, not a
separately optimized local policy.

Only optimal/locally optimal solver terminations produce actions. This first
interface does not certify AC feasibility, SOCP tightness, or replay equality;
use the returned summary and an OpenDSS replay to accept training labels.

## Per-unit training pairs

`PUNormalizer` takes `s_base_kva` and `v_base_kv`, loaded from the `base`
object in `config.json`. Both repositories import the same lightweight package:

```json
"base": {
  "s_base_kva": 1000.0,
  "v_base_kv": 11.0
}
```

Every active or reactive power is divided by `s_base_kva`, including loads,
PV availability, previous device powers and the teacher's actions. The same
total network base is used for each phase and for aggregate powers. For
example, -20 kW becomes -0.02 pu with a 1000 kVA base. Values above 1 pu and
negative values are preserved.

Voltage observations already use `v_before_pu` and are copied directly, without
a second conversion or centering around 1. The single-phase environment uses
the configured voltage base; three-phase observations use the OpenDSS base at
each bus, including the appropriate voltage levels across transformers. The
normalizer preserves this voltage convention. SoC, timestamps, interval length,
angles and tariffs also keep their original units. Currents are not part of
the current observation contract.

```python
from pu_normalizer import PUNormalizer
from opf_teacher.normalization import TimeEncoder

normalizer = PUNormalizer(case.base.s_base_kva, case.base.v_base_kv)
time_encoder = TimeEncoder()
state, action = teacher.state_action()  # Already in pu.
state["time"] = time_encoder.encode(state.pop("timestamp"))
pair = history.append(state, action)
observation, reward, terminated, truncated, info = env.step(action)

# Optional physical powers for inspection or reporting.
physical_info = normalizer.denormalize(info)
```

The conversion creates independent copies and renames power fields:

| Physical field | Per-unit field |
|---|---|
| `p_load_kw`, `q_load_kvar` | `p_load_pu`, `q_load_pu` |
| `previous_p_kw`, `previous_q_kvar` | `previous_p_pu`, `previous_q_pu` |
| `available_kw`, `generation_kw` | `available_pu`, `generation_pu` |
| `p_net_kw`, `q_injection_kvar` | `p_net_pu`, `q_injection_pu` |

`normalize()` and `denormalize()` support the scalar single-phase and
phase-indexed three-phase dictionaries, including local pairs. They use fixed
bases for every position in the temporal window and do not fit dataset
statistics. The internal `Case` and its typed states/results retain explicit
physical field names; `Case.state_action()` is the physical model interface.
Use `Teacher.state_action()` for exchange with the environment and NN.
`normalizer.as_dict()` provides JSON-serializable normalization metadata.
`PerUnitNormalizer(case.base)` remains only as a compatibility constructor.

## Timestamp features for neural networks

`TimeEncoder` converts the time of day to two numeric features:

```text
angle = 2 * pi * hours_since_midnight / 24
hour_sin = sin(angle)
hour_cos = cos(angle)
```

The two features are between -1 and 1 and keep neighboring times close across
midnight. Minutes and fractional seconds contribute to the angle. The encoder
is a fixed transformation and has no learned parameters or training step.

| Time | `hour_sin` | `hour_cos` |
|---|---|---|
| 00:00 | 0 | 1 |
| 06:00 | 1 | 0 |
| 12:00 | 0 | -1 |
| 18:00 | -1 | 0 |

The training scripts replace each state's timestamp string with these features
before adding it to `StateActionHistory`. The original timestamp remains in
the row metadata, and the named teacher observation keeps its timestamp.
Select the `x.*` numeric columns as the network inputs and `y.*` as the targets.

The same encoder is available for single-phase, three-phase and local states.
For a local state, add `state["time"] = time_encoder.encode(timestamp)` using
the timestamp of that observation. The encoding uses the timestamp's local
clock, without converting timezone offsets to UTC. It represents time of day;
weekday and annual seasonality are not included. `time_encoder.as_dict()`
records the method, period and feature names in the run JSON.

## Case ownership and solver settings

When given a `Case`, `Teacher` creates a deep copy. Observations, results and
actions modify only `teacher.case`; `solve()` returns that owned case. Two
teachers built from the same input case are independent. Callers that previously
read results from the input case must now use the returned case or `teacher.case`.
Paths also load a private case for each teacher.

Use a settings object for explicit solver configuration:

```python
from opf_teacher.teacher import SolveConfig, Teacher

teacher = Teacher("examples/case5", formulation="single_phase_active")
teacher.build()
settings = SolveConfig(solver="appsi_highs")
solved_case = teacher.solve(config=settings)
```

`SolveConfig` includes `tee`, `socp_gap_tolerance`, `maxiter`, `ftol` and
`solver_options` for Pyomo solver options. Legacy calls such as
`solve(solver="appsi_highs")` and `solve(maxiter=500)` remain supported, but cannot
be combined with `config`.

Single-phase and three-phase cases have distinct contracts:

| Module | Load profile | Device representation |
|---|---|---|
| `opf_teacher.opf.single_phase.components` / `opf_teacher.opf.single_phase.states` | One series per bus quantity | BESS/PV directly |
| `opf_teacher.opf.three_phase.components` / `opf_teacher.opf.three_phase.states` | One series per phase | BESS/PV with an explicit connection |

Use `ThreePhaseTeacher` for phase-indexed observations and actions. A supplied
case must match the selected formulation. The common `Teacher` entry point
preserves compatibility; the model builders, solver methods and action extraction
remain separate for each contract.

## State-action history

`Teacher` solves the OPF horizon. `StateActionHistory` collects observed states
and pairs them with the current action. It accepts central or local dictionaries.
Use one instance per episode and controller; local devices need separate histories.

The default `window=1` returns `([current_state], current_action)` immediately.
Larger windows return `([past_states, ..., current_state], current_action)` after
collecting enough observations. The state input is always a list, even with
window one; the action target always belongs to the current step.

```python
from opf_teacher.teacher import StateActionHistory

history = StateActionHistory(window=3)  # Omit window for a single-state pair.

# At each observed step, after solving the corresponding horizon:
state, action = teacher.state_action()
# Optionally select a local pair with PairSelector before appending.
pair = history.append(state, action)
if pair is not None:
    states, current_action = pair
    # states = [state_t_minus_2, state_t_minus_1, state_t]

# Before starting another episode:
history.reset()
```

The first `window - 1` calls return `None`. Returned pairs are independent
copies. The caller supplies observations in chronological order at the chosen
sampling interval; the class does not check timestamps or advance the environment.
Do not fill the window with the OPF's predicted future trajectory.

`teacher.state_action()` returns a single `(state, action)` pair in pu.
`TemporalTeacher` remains as a compatibility name with its previous default
`window=3`. New code should use `StateActionHistory`, whose default is one.

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

The single-phase formulations are assembled by `ActivePowerModel` and
`DistFlowModel`. Each class keeps the physical `Case`, defines its variables,
and registers named rules for network, battery and PV constraints. Dictionaries
index component objects by ID; parameters are read from the components instead
of copied into parallel dictionaries. Legacy `build_active_power_model()` and
`build_model()` functions remain compatibility entry points.

The three-phase `IVRModel` evaluates candidate dispatches into a
`DispatchEvaluation` object with named cost, voltage, current, grid power, SoC
and residual fields. Array indices and SciPy solver arguments remain internal.

```text
opf_teacher/
|-- __init__.py         public teacher classes
|-- teacher.py          teacher and state-action history
|-- normalization.py    per-unit conversion and time encoding
`-- opf/
    |-- common.py       shared components, schema versions, case paths and PV curves
    |-- formulations.py formulation selection
    |-- opendss.py      shared static network importer
    |-- single_phase/
    |   |-- components.py   single-phase network, case and result classes
    |   |-- data.py         single-phase case loader
    |   |-- states.py       single-phase observations and actions
    |   |-- active_model.py active-power LinDistFlow
    |   |-- distflow_socp.py DistFlow SOCP, including fixed power factor
    |   |-- pv_droop.py     Volt-VAr and Volt-Watt equations
    |   |-- results.py      SOCP results and relaxation gap
    |   `-- export.py       single-phase CSV export
    `-- three_phase/
        |-- components.py   phase-indexed network and device connections
        |-- data.py         three-phase case loader
        |-- states.py       phase-indexed observations and actions
        |-- phases.py       phase names and validation
        |-- ivr_model.py    unbalanced AC-IVR
        |-- results.py      three-phase results
        `-- export.py       three-phase CSV export
main.py                 battery OPF and CSV results
train-w1.py             OpenDSS X/Y pairs, window 1
train-w3.py             OpenDSS X/Y pairs, window 3
```

Use explicit formulation imports, for example
`from opf_teacher.opf.single_phase.data import load_case` and
`from opf_teacher.opf.three_phase.export import write_result_csvs`. The `opf_teacher.opf` package exposes
only shared types and formulation names. The old root-level forwarding modules
have been removed; external callers must update those import paths.

Single-phase local pairs are built with
`opf_teacher.opf.single_phase.states.local_state_action(device, bus)`. Shared BESS/PV
objects only store data and do not import either formulation. Three-phase local
pairs remain on `ConnectedBess` and `ConnectedPv`.

Case path resolution and schema versions are defined in `opf_teacher.opf.common`.
The former `case_source` and `contracts` modules have been removed. Formulation
selection accepts only `single_phase_active`, `single_phase_socp` and
`three_phase_ivr`; aliases and `active_power_only` have been removed.
