"""Radial series-impedance snapshots with fixed complex power consumption."""


def bus_voltages(case, power_kw):
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
            return {bus: abs(value) for bus, value in voltage.items()}
    raise RuntimeError("Pre-action power flow did not converge")
