from __future__ import annotations

import argparse
from pathlib import Path

from opf.components import Case
from teacher import BessOpt


def _aggregate(devices, attr):
    """Soma uma serie de resultado sobre todos os dispositivos (1 ou varios)."""
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
    fig.suptitle(f"Operacao otima — {case.name}", fontsize=14, fontweight="bold")

    # --- 1) Tarifa x importacao ---------------------------------------- #
    ax = axes[0]
    ax.step(t, grid.import_kw, where="mid", color="tab:blue", label="Import rede (kW)")
    if grid.export_kw.max() > 1.0:
        ax.step(t, -grid.export_kw, where="mid", color="tab:cyan", label="Export rede (kW)")
    ax.set_ylabel("Potencia (kW)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    axp = ax.twinx()
    axp.step(t, case.price, where="mid", color="tab:red", lw=2, label="Tarifa")
    axp.set_ylabel("Tarifa (R$/kWh)", color="tab:red")
    axp.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title("Tarifa x importacao da rede")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 2) Bateria: carga/descarga x SoC ------------------------------ #
    ax = axes[1]
    if case.bess:
        p_net = _aggregate(case.bess, "p_net_kw")   # >0 carrega, <0 descarrega
        soc = _aggregate(case.bess, "soc_kwh")
        colors = ["tab:orange" if v >= 0 else "tab:green" for v in p_net]
        ax.bar(t, p_net, width=0.03, color=colors, label="Carga (+) / Descarga (-)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("Potencia BESS (kW)")
        axs = ax.twinx()
        axs.plot(t, soc, color="tab:purple", lw=2, marker=".", label="SoC")
        axs.set_ylabel("SoC (kWh)", color="tab:purple")
        axs.tick_params(axis="y", labelcolor="tab:purple")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Bateria: despacho x estado de carga")
    ax.grid(True, alpha=0.3)

    # --- 3) PV disponivel x gerado ------------------------------------- #
    ax = axes[2]
    if case.pv:
        avail = _aggregate(case.pv, "avail_kw")
        gen = _aggregate(case.pv, "gen_kw")
        ax.fill_between(t, avail, color="gold", alpha=0.4, label="PV disponivel")
        ax.plot(t, gen, color="darkorange", lw=2, label="PV gerado")
    ax.plot(t, load_total, color="tab:gray", lw=1.5, ls="--", label="Carga total")
    ax.set_ylabel("Potencia (kW)")
    ax.set_title("Geracao PV x carga")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 4) Tensoes por barra ------------------------------------------ #
    ax = axes[3]
    for b, bus in case.buses.items():
        ax.plot(t, bus.result.v_pu, lw=1.4, marker=".", ms=3, label=f"V{b}")
    vmin = min(b.v_min_pu for b in case.buses.values())
    vmax = max(b.v_max_pu for b in case.buses.values())
    ax.axhline(vmin, color="r", ls=":", lw=1, label=f"limite {vmin:g}")
    ax.axhline(vmax, color="r", ls=":", lw=1)
    ax.set_ylabel("Tensao (p.u.)")
    ax.set_title("Tensoes por barra")
    ax.legend(loc="lower left", fontsize=8, ncol=len(case.buses) + 1)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"figura salva em: {out_path}")
    if show:
        plt.show()


def print_summary(case: Case):
    s = case.summary
    print(f"\n=== {case.name} - status: {s.status} ===")
    for k, v in s.as_dict().items():
        if k != "status":
            print(f"  {k:>20}: {v:,.3f}")


def main():
    ap = argparse.ArgumentParser(description="Resolve um exemplo e plota o despacho otimo.")
    ap.add_argument("example", nargs="?", default="examples/case5", help="diretorio do exemplo")
    ap.add_argument("--solver", default="gurobi_direct")
    ap.add_argument("--out", default="figures", help="diretorio de saida das figuras")
    ap.add_argument("--show", action="store_true", help="abrir janela do matplotlib")
    args = ap.parse_args()

    case = BessOpt(args.example).build().solve(solver=args.solver)
    print_summary(case)
    plot_results(case, Path(args.out) / f"{Path(args.example).name}_dispatch.png", args.show)


if __name__ == "__main__":
    main()
