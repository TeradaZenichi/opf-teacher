"""Radial series-impedance snapshots with fixed complex power consumption."""
import cmath
import math

from opf_teacher.opf.single_phase.components import phase_count, system_voltage_base_kv
from opf_teacher.opf.single_phase.states import BusState


def pre_action_buses(case, bess, pv, measurements=None):
    """Observe current demand with held commands, without integrating energy."""
    buses = {bid: BusState(case.grid.v_ref_pu, float(bus.p_load_kw.iloc[0]),
                           float(bus.q_load_kw.iloc[0])) for bid, bus in case.buses.items()}
    power = {bid: complex(state.p_load_kw, state.q_load_kvar) for bid, state in buses.items()}
    for device in case.bess:
        state = bess[device.id]
        power[device.bus] += complex(state.previous_p_kw, -state.previous_q_kvar)
    for device in case.pv:
        state = pv[device.id]
        power[device.bus] -= complex(min(state.previous_p_kw, state.available_kw), state.previous_q_kvar)
    for bid, voltage in bus_voltages(case, power, measurements).items():
        buses[bid].v_before_pu = voltage
    return buses


def bus_voltages(case, power_kw, measurements=None):
    """Return pu voltage magnitudes; power_kw contains kW + j kvar by bus."""
    order = [case.root]
    for bus in order:
        order.extend(case.children[bus])
    branches = {branch.to_bus: branch for branch in case.branches}
    voltage = dict.fromkeys(order, complex(case.grid.v_ref_pu))
    power = {bus: power_kw[bus] / case.base.s_base_kva for bus in order}
    for _ in range(100):
        previous = voltage.copy()
        current = {bus: (power[bus] / voltage[bus]).conjugate() for bus in order}
        for bus in reversed(order[1:]):
            branch = branches[bus]
            current[bus] *= branch.tap_ratio
            current[branch.from_bus] += current[bus]
        for bus in order[1:]:
            branch = branches[bus]
            parent = branch.from_bus
            r, x = branch.impedance_pu(case.base, case.buses[parent])
            voltage[bus] = branch.tap_ratio * (voltage[parent] - complex(r, x) * current[bus])
        if max(abs(voltage[bus] - previous[bus]) for bus in order) < 1e-10:
            if measurements is not None:
                measurements["angles"] = {bus: math.degrees(cmath.phase(value)) for bus, value in voltage.items()}
                measurements["branches"] = []
                current = {bus: (power[bus] / voltage[bus]).conjugate() for bus in order}
                for bus in reversed(order[1:]):
                    current[bus] *= branches[bus].tap_ratio
                    current[branches[bus].from_bus] += current[bus]
                for branch in case.branches:
                    if phase_count(branch.phases) != 1:
                        raise ValueError("Branch extras require a single-phase equivalent")
                    parent, child = branch.from_bus, branch.to_bus
                    primary = current[child]
                    for terminal, bus, amps_pu in ((1, parent, primary), (2, child, -primary / branch.tap_ratio)):
                        power_pu = voltage[bus] * amps_pu.conjugate()
                        kv = system_voltage_base_kv(case.buses[bus].kv_base_ln, branch.phases, case.base.v_base_kv)
                        measurements["branches"].append({
                            "element": f"{branch.element_type}.{branch.name}", "terminal": terminal,
                            "bus": bus, "phase": "a", "p_flow_pu": power_pu.real, "q_flow_pu": power_pu.imag,
                            "current_a": abs(amps_pu) * case.base.s_base_kva / kv,
                            "current_angle_deg": math.degrees(cmath.phase(amps_pu)),
                        })
            return {bus: abs(value) for bus, value in voltage.items()}
    raise RuntimeError("Pre-action power flow did not converge")
