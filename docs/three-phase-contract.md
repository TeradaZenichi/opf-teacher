# Unbalanced three-phase contract

The `opf.three_phase` package is independent from the existing single-phase
equivalent formulations. It defines the data read by the AC-IVR model.

## Phase names

Public data uses `a`, `b`, and `c`. OpenDSS nodes `1`, `2`, and `3` are accepted
at construction time and normalized to those names. Phase sets are explicit;
missing phases are not silently filled with zeros.

## Network data

`Bus` stores active and reactive load profiles as one `pandas.Series` per phase.
All phase profiles must use the case timestamp index. Aggregate load properties
are provided only for reporting and compatibility.

`Branch` stores the complete complex series-impedance matrix in the exact order
given by `phases`. Off-diagonal entries represent mutual coupling. Ampacity can
be scalar or phase-specific. A full complex tap matrix is also supported by the
contract. The solver uses this matrix for lines and two-winding `wye-wye` and
`delta-wye` transformers, including fixed taps, transformer phase shift, and
different voltage bases at each bus. The receiving `wye` winding is treated as
grounded.

`Grid` requires voltage magnitude and angle references per phase. `Case`
validates bus, branch, grid, device, timestamp, and phase consistency before a
model is built.

## Devices and signs

`DeviceConnection` declares the bus, phases, `wye` or `delta` connection, and
whether dispatch is aggregate or per phase. It wraps the existing BESS and PV
technical parameters without changing their established signs. A connected PV
also carries availability forecasts by phase, whose sum must equal its existing
aggregate profile:

- BESS `p_net_kw > 0` means charging;
- BESS `p_net_kw < 0` means discharging;
- positive reactive power means injection;
- PV generation is positive.

Actions expose phase values and aggregate-total properties. BESS SoC remains a
single shared energy state. Aggregate dispatch is balanced among the connected
phases. Per-phase dispatch keeps independent phase commands while enforcing the
inverter rating, aggregate active-power limits, and shared SoC.

For a three-phase `delta` device, public keys identify OpenDSS winding legs:
`a = a-c`, `b = b-a`, and `c = c-b`. Volt-VAr and Volt-Watt use the
corresponding line-to-line voltage divided by its nominal line-to-line base.

## Current support

The three-phase data classes, case loader, nonlinear AC-IVR power flow,
multiple time-coupled BESS, aggregate and per-phase dispatch, `wye` and `delta`
devices, reactive BESS operation, controllable PV, Volt-VAr/Volt-Watt curves,
transformer taps and phase shift, result attachment, and CSV exports are
implemented. SciPy SLSQP therefore reports a local,
physically feasible solution rather than a global-optimality certificate.

The following are not supported:

- explicit neutral-conductor modeling.
- ungrounded `wye-delta` and `delta-delta` transformer secondaries;
- single-phase transformer banks and transformers with more than two windings;
- SDP certification of the local solution.

Use `Teacher(path, formulation="three_phase_ivr").solve()` for a phase-indexed
case. A legacy case with aggregate demand columns is rejected by the
three-phase loader instead of silently solving the single-phase equivalent.
