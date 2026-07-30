"""Resolve um OPF com controle ativo de BESS/PV e exporta os resultados."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from opf.active_model import solve_active_power_opf
from opf.data import load_case
from opf.export import (
    write_result_csvs,
)


DEFAULT_CASE_PATH = PROJECT_ROOT / "examples" / "case5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--solver", default="appsi_highs")
    parser.add_argument("--tie-break-penalty", type=float, default=1e-6)
    return parser.parse_args()


def main():
    args = parse_args()
    solved = solve_active_power_opf(
        load_case(args.case),
        solver=args.solver,
        tie_break_penalty=args.tie_break_penalty,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{solved.name}_active_power_opf"
    paths = write_result_csvs(solved, args.output_dir, stem=stem)

    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"status: {solved.summary.status}")
    print(f"objective_cost: {solved.summary.objective_cost:.6f}")
    print("formulation: active-power LinDistFlow")


if __name__ == "__main__":
    main()
