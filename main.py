"""Solve the battery OPF and export the predicted trajectories."""
from pathlib import Path

from opf_teacher.opf.single_phase.export import write_result_csvs
from opf_teacher.teacher import Teacher


# Change the run settings here.
ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "examples" / "case5_local"  # Input scenario.
RESULTS_PATH = ROOT / "results"
FORMULATION = "single_phase_active"  # Or single_phase_socp.
OUTPUT_NAME = "battery_opf"  # CSV filename prefix.


def main():
    teacher = Teacher(CASE_PATH, formulation=FORMULATION)
    teacher.build()
    solved = teacher.solve()
    paths = write_result_csvs(solved, RESULTS_PATH, stem=OUTPUT_NAME)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return solved


if __name__ == "__main__":
    main()
