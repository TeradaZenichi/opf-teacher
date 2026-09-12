"""Solve and export the demand-unbalanced three-phase example."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opf.three_phase.export import write_result_csvs
from teacher import Teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", nargs="?", default=str(ROOT / "examples" / "case5_unbalanced"))
    parser.add_argument("--output", default=str(ROOT / "results"))
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args()

    solved = Teacher(args.case, formulation="three_phase_ivr").solve(
        solver="scipy_slsqp",
        tee=args.tee,
        maxiter=args.maxiter,
    )
    paths = write_result_csvs(solved, args.output)
    print(solved.summary.as_dict())
    print(solved.quality)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return solved


if __name__ == "__main__":
    main()
