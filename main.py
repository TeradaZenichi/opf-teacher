from __future__ import annotations

from pathlib import Path

from opf.components import Case
from teacher import BessOpt


PROJECT_ROOT = Path(__file__).resolve().parent
CASE_PATH = PROJECT_ROOT / "examples" / "case5"
OUTPUT_PATH = PROJECT_ROOT / "figures" / "case5_dispatch.png"
SOLVER = "gurobi_direct"
SOCP_GAP_TOLERANCE = 1e-5
SHOW_PLOT = False


def _aggregate(devices, attr):
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

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
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
        p_net = _aggregate(case.bess, "p_net_kw")
        q_bess = _aggregate(case.bess, "q_kvar")
        soc = _aggregate(case.bess, "soc_kwh")
        charge = p_net.clip(lower=0.0)
        discharge = p_net.clip(upper=0.0)
        ax.bar(t, charge, width=0.03, color="tab:orange", label="P charge (+)")
        ax.bar(t, discharge, width=0.03, color="tab:green", label="P discharge (-)")
        ax.step(t, q_bess, where="mid", color="tab:blue", lw=2,
                label="Q: injection (+) / absorption (-)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("BESS power (kW / kVAr)")
        axs = ax.twinx()
        axs.plot(t, soc, color="tab:purple", lw=2, marker=".", label="SoC")
        axs.set_ylabel("SoC (kWh)", color="tab:purple")
        axs.tick_params(axis="y", labelcolor="tab:purple")
        handles, labels = ax.get_legend_handles_labels()
        soc_handles, soc_labels = axs.get_legend_handles_labels()
        ax.legend(handles + soc_handles, labels + soc_labels,
                  loc="upper left", fontsize=8)
    ax.set_title("BESS optimal active and reactive power")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if case.pv:
        avail = _aggregate(case.pv, "avail_kw")
        gen = _aggregate(case.pv, "gen_kw")
        q_pv = _aggregate(case.pv, "q_kvar")
        ax.fill_between(t, avail, color="gold", alpha=0.4, label="PV available")
        ax.step(t, gen, where="mid", color="darkorange", lw=2,
                label="PV optimal P")
        ax.step(t, q_pv, where="mid", color="tab:blue", lw=2,
                label="PV optimal Q")
    ax.plot(t, load_total, color="tab:gray", lw=1.5, ls="--", label="Total load")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Power (kW / kVAr)")
    ax.set_title("PV optimal active and reactive power")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    for b, bus in case.buses.items():
        ax.step(t, bus.result.v_pu, where="mid", lw=1.4, label=f"V{b}")
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
            if isinstance(v, bool):
                value = str(v)
            else:
                value = f"{v:,.6g}" if isinstance(v, (int, float)) else str(v)
            print(f"  {k:>28}: {value}")


def main():
    case = BessOpt(CASE_PATH).build().solve(
        solver=SOLVER,
        socp_gap_tolerance=SOCP_GAP_TOLERANCE,
    )
    print_summary(case)
    plot_results(case, OUTPUT_PATH, SHOW_PLOT)


if __name__ == "__main__":
    main()
