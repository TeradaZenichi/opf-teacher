"""Generate X/Y windows from one optimized trajectory."""
import json
from pathlib import Path

import pandas as pd

from opf_teacher.normalization import PUNormalizer, TimeEncoder
from opf_teacher import PairConfig, PairSelector, Teacher, collect_pairs


# Change the run settings here.
ROOT = Path(__file__).resolve().parents[2]
CASE_PATH = ROOT / "examples" / "case5_local"  # Input scenario.
CSV_PATH = ROOT / "results" / "train.csv"
# single_phase_active: active-power single-phase equivalent.
# single_phase_socp: single-phase P/Q model with SOCP relaxation.
# three_phase_ivr: unbalanced three-phase model with phase-specific values.
FORMULATION = "single_phase_active"
# central: all buses/devices in X and all device actions in Y.
# local: selected BESS's bus/state in X and only its action in Y.
# Both modes include encoded time and use the same global OPF solution.
OBSERVATION_MODE = "central"
DEVICE_ID = "b1"  # BESS selected in local mode; ignored in central mode.
WINDOW = 3  # Number of observed states in each X.


def main(window=WINDOW, output=CSV_PATH):
    output = Path(output)
    original_case = Teacher(CASE_PATH, formulation=FORMULATION).case
    config = PairConfig(formulation=FORMULATION, mode=OBSERVATION_MODE, device_id=DEVICE_ID, window=window)
    x, y = collect_pairs(original_case, config)
    selector = PairSelector(original_case, OBSERVATION_MODE, DEVICE_ID)
    normalizer = PUNormalizer(original_case.base.s_base_kva, original_case.base.v_base_kv)
    time_encoder = TimeEncoder()
    phase_order = x[0][-1].get("phase_order", ["a"])
    rows = []
    for history, action in zip(x, y):
        timestamp = history[-1]["timestamp"]
        for state in history:
            state["time"] = time_encoder.encode(state.pop("timestamp"))
            state.pop("phase_order", None)
        rows.append({"timestamp": timestamp, "x": {str(i): state for i, state in enumerate(history)}, "y": action})

    table = pd.json_normalize(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    metadata = {
        "case": original_case.name, "formulation": FORMULATION,
        "observation": selector.as_dict(),
        "window": window, "periods": original_case.n_periods, "pairs": len(table),
        "phase_order": phase_order, "teacher_mode": "single_optimization",
        "normalization": normalizer.as_dict(), "time_encoding": time_encoder.as_dict(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{len(table)} pairs saved to: {output} (window={window})")
    return table


if __name__ == "__main__":
    main(window=WINDOW, output=CSV_PATH)
