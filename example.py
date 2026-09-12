"""First-interval X/Y pairs for central and local battery control."""
from pathlib import Path
from pprint import pprint

from opf import BessState, BusState, PvState
from teacher import Teacher, TemporalTeacher


EXAMPLES = Path(__file__).resolve().parent / "examples"
CENTRAL_CASE = EXAMPLES / "case5_central"
LOCAL_CASE = EXAMPLES / "case5_local"
WINDOW = 3
STEPS = 5


def solve_case(path, step=0):
    teacher = Teacher(path, active_power_only=True)
    case = teacher.case
    if not 0 <= step < case.n_periods:
        raise ValueError("step must be inside the case horizon")

    case.timestamps = case.timestamps[step:]
    case.price = case.price.iloc[step:].copy()
    for bus in case.buses.values():
        bus.p_load_kw = bus.p_load_kw.iloc[step:].copy()
        bus.q_load_kw = bus.q_load_kw.iloc[step:].copy()
    for device in case.pv:
        device.avail_kw = device.avail_kw.iloc[step:].copy()

    # Synthetic idle history: constant SoC, zero previous commands, illustrative 1 pu voltage.
    # Teacher actions are labels only; they are not applied to this observation sequence.
    buses = {
        bid: BusState(1.0, float(bus.p_load_kw.iloc[0]), float(bus.q_load_kw.iloc[0]))
        for bid, bus in case.buses.items()
    }
    bess = {d.id: BessState(d.soc_init_frac, 0.0, 0.0) for d in case.bess}
    pv = {d.id: PvState(float(d.avail_kw.iloc[0]), 0.0, 0.0) for d in case.pv}
    teacher.observe(buses=buses, bess=bess, pv=pv)
    return teacher.solve()  # HiGHS, active power only; device Q is zero.


def main():
    central_history = TemporalTeacher(window=WINDOW)
    local_history = TemporalTeacher(window=WINDOW)
    if STEPS < WINDOW:
        raise ValueError("STEPS must be at least WINDOW to show a complete pair")

    for step in range(STEPS):
        central_case = solve_case(CENTRAL_CASE, step)
        x_central, y_central = central_case.state_action()
        central_pair = central_history.append(x_central, y_central)

        local_case = solve_case(LOCAL_CASE, step)
        battery = local_case.bess[0]
        x_local, y_local = battery.state_action(local_case.buses[battery.bus])
        local_pair = local_history.append(x_local, y_local)

        # Breakpoint: pairs are None during warm-up, then (past/current X, current Y).
        status = "ready" if central_pair is not None else "collecting history"
        print(f"{central_case.index[0]} | {status}")

    x_central_history, y_central_current = central_pair
    x_local_history, y_local_current = local_pair
    pprint({"central": {"x": x_central_history, "y": y_central_current},
            "local": {"x": x_local_history, "y": y_local_current}}, sort_dicts=False)
    return central_pair, local_pair


if __name__ == "__main__":
    main()
