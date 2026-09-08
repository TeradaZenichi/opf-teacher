"""First-interval X/Y pairs for central and local battery control."""
from pathlib import Path
from pprint import pprint

from opf import BessState, BusState, PvState
from teacher import Teacher


EXAMPLES = Path(__file__).resolve().parent / "examples"
CENTRAL_CASE = EXAMPLES / "case5_central"
LOCAL_CASE = EXAMPLES / "case5_local"


def solve_case(path):
    teacher = Teacher(path, active_power_only=True)
    case = teacher.case

    # Illustrative observations; real pre-action voltage must come from the environment.
    buses = {
        bid: BusState(1.0, float(bus.p_load_kw.iloc[0]), float(bus.q_load_kw.iloc[0]))
        for bid, bus in case.buses.items()
    }
    bess = {d.id: BessState(d.soc_init_frac, 0.0, 0.0) for d in case.bess}
    pv = {d.id: PvState(float(d.avail_kw.iloc[0]), 0.0, 0.0) for d in case.pv}
    teacher.observe(buses=buses, bess=bess, pv=pv)
    return teacher.solve()  # HiGHS, active power only; device Q is zero.


def main():
    central_case = solve_case(CENTRAL_CASE)
    x_central, y_central = central_case.state_action()

    local_case = solve_case(LOCAL_CASE)
    battery = local_case.bess[0]
    x_local, y_local = battery.state_action(local_case.buses[battery.bus])

    # Breakpoint: compare all buses/devices in central X/Y with just b1 and its bus in local X/Y.
    print(f"Central: {central_case.summary.status} | Local: {local_case.summary.status}")
    pprint({"central": {"x": x_central, "y": y_central},
            "local": {"x": x_local, "y": y_local}}, sort_dicts=False)

    # Each pair uses its case's first action. Both teachers optimize the full network.


if __name__ == "__main__":
    main()
