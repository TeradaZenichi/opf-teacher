from __future__ import annotations

import argparse
from pathlib import Path

from opf.components import Case
from teacher import BessOpt


def _aggregate(devices, attr):
    """Sum a result series over all devices (one or many)."""
    total = None
    for d in devices:
        s = getattr(d.result, attr)
        total = s.copy() if total is None else total + s
    return total


def plot_results(case: Case, out_path: Path, show: bool):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    t = case.index
    grid = case.grid.result
    load_total = sum(bus.p_load_kw for bus in case.buses.values())

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(f"Optimal operation — {case.name}", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.step(t, grid.import_kw, where="mid", color="tab:blue", label="Grid import (kW)")
    if grid.export_kw.max() > 1.0:
        ax.step(t, -grid.export_kw, where="mid", color="tab:cyan", label="Grid export (kW)")
    ax.set_ylabel("Power (kW)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    axp = ax.twinx()
    axp.step(t, case.price, where="mid", color="tab:red", lw=2, label="Price")
    axp.set_ylabel("Price (R$/kWh)", color="tab:red")
    axp.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title("Price vs grid import")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if case.bess:
        p_net = _aggregate(case.bess, "p_net_kw")   # >0 charge, <0 discharge
        soc = _aggregate(case.bess, "soc_kwh")
        colors = ["tab:orange" if v >= 0 else "tab:green" for v in p_net]
        ax.bar(t, p_net, width=0.03, color=colors, label="Charge (+) / Discharge (-)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("BESS power (kW)")
        axs = ax.twinx()
        axs.plot(t, soc, color="tab:purple", lw=2, marker=".", label="SoC")
        axs.set_ylabel("SoC (kWh)", color="tab:purple")
        axs.tick_params(axis="y", labelcolor="tab:purple")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Battery: dispatch vs state of charge")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if case.pv:
        avail = _aggregate(case.pv, "avail_kw")
        gen = _aggregate(case.pv, "gen_kw")
        ax.fill_between(t, avail, color="gold", alpha=0.4, label="PV available")
        ax.plot(t, gen, color="darkorange", lw=2, label="PV generated")
    ax.plot(t, load_total, color="tab:gray", lw=1.5, ls="--", label="Total load")
    ax.set_ylabel("Power (kW)")
    ax.set_title("PV generation vs load")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    for b, bus in case.buses.items():
        ax.plot(t, bus.result.v_pu, lw=1.4, marker=".", ms=3, label=f"V{b}")
    vmin = min(b.v_min_pu for b in case.buses.values())
    vmax = max(b.v_max_pu for b in case.buses.values())
    ax.axhline(vmin, color="r", ls=":", lw=1, label=f"limit {vmin:g}")
    ax.axhline(vmax, color="r", ls=":", lw=1)
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title("Voltages per bus")
    ax.legend(loc="lower left", fontsize=8, ncol=len(case.buses) + 1)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"figure saved to: {out_path}")
    if show:
        plt.show()


def print_summary(case: Case):
    s = case.summary
    print(f"\n=== {case.name} - status: {s.status} ===")
    for k, v in s.as_dict().items():
        if k != "status":
            print(f"  {k:>20}: {v:,.3f}")


def main():
    ap = argparse.ArgumentParser(description="Solve an example and plot the optimal dispatch.")
    ap.add_argument("example", nargs="?", default="examples/case5", help="example directory")
    ap.add_argument("--solver", default="gurobi_direct")
    ap.add_argument("--out", default="figures", help="output directory for figures")
    ap.add_argument("--show", action="store_true", help="open the matplotlib window")
    args = ap.parse_args()

    case = BessOpt(args.example).build().solve(solver=args.solver)
    print_summary(case)
    plot_results(case, Path(args.out) / f"{Path(args.example).name}_dispatch.png", args.show)


if __name__ == "__main__":
    main()
