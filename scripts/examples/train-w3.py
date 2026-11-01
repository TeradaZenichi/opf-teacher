"""Generate chronological X/Y pairs from opf_teacher.teacher actions executed in OpenDSS."""
import json
from pathlib import Path

import pandas as pd
from opendss_environment import EnvironmentConfig, MicrogridEnv

from opf_teacher.opf.single_phase import load_case
from opf_teacher.normalization import PUNormalizer, TimeEncoder
from opf_teacher.teacher import StateActionHistory, Teacher


# Change the run settings here.
ROOT = Path(__file__).resolve().parents[2]
CASE_PATH = ROOT / "examples" / "case5_local"  # Input scenario.
CSV_PATH = ROOT / "results" / "train-w3.csv"
FORMULATION = "single_phase_active"  # Or single_phase_socp.
WINDOW = 3  # Number of observed states in each X.


def main():
    original_case = load_case(CASE_PATH)
    normalizer = PUNormalizer(original_case.base.s_base_kva, original_case.base.v_base_kv)
    time_encoder = TimeEncoder()
    history = StateActionHistory(window=WINDOW)
    env = MicrogridEnv(EnvironmentConfig(CASE_PATH, original_case.n_periods))
    rows = []
    observation, _ = env.reset()
    for step in original_case.periods:
        horizon = original_case.horizon(start=step)
        teacher = Teacher(horizon, formulation=FORMULATION)
        teacher.observe_dict(observation)
        teacher.build()
        solved = teacher.solve()
        state, action = teacher.state_action()
        state["time"] = time_encoder.encode(state.pop("timestamp"))
        pair = history.append(state, action)
        observation, _, terminated, truncated, _ = env.step(action)

        label_valid = solved.quality is None or solved.quality.get("valid_for_imitation", True)
        if pair is not None and label_valid:
            states, current_action = pair
            state_history = {}
            for position, observed_state in enumerate(states):
                state_history[str(position)] = observed_state
            rows.append({"timestamp": solved.index[0].isoformat(), "x": state_history, "y": current_action})
        if terminated or truncated:
            break
    env.close()

    table = pd.json_normalize(rows)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    metadata = {
        "case": original_case.name, "window": WINDOW,
        "normalization": normalizer.as_dict(), "time_encoding": time_encoder.as_dict(),
    }
    CSV_PATH.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{len(table)} pairs saved to: {CSV_PATH}")
    return table


if __name__ == "__main__":
    main()
