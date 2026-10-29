"""Generate chronological X/Y pairs from teacher actions executed in OpenDSS."""
import json
from pathlib import Path

import pandas as pd
from opendss_env import EnvironmentConfig, MicrogridEnv

from opf.single_phase import load_case
from normalization import PerUnitNormalizer, TimeEncoder
from teacher import StateActionHistory, Teacher


# Change the run settings here.
ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "examples" / "case5_local"  # Input scenario.
CSV_PATH = ROOT / "results" / "train-w1.csv"
FORMULATION = "single_phase_active"  # Or single_phase_socp.
WINDOW = 1  # Number of observed states in each X.


def main():
    original_case = load_case(CASE_PATH)
    normalizer = PerUnitNormalizer(original_case.base)
    time_encoder = TimeEncoder()
    history = StateActionHistory(window=WINDOW)
    env = MicrogridEnv(EnvironmentConfig(CASE_PATH, original_case.n_periods, ()))
    rows = []
    env.reset()
    for step in original_case.periods:
        horizon = original_case.horizon(start=step)
        teacher = Teacher(horizon, formulation=FORMULATION)
        teacher.observe_dict(env.observe())
        teacher.build()
        solved = teacher.solve()
        state, action = solved.state_action()
        normalized_state = normalizer.normalize(state)
        normalized_state["time"] = time_encoder.encode(normalized_state.pop("timestamp"))
        normalized_action = normalizer.normalize(action)
        pair = history.append(normalized_state, normalized_action)
        _, _, terminated, truncated, _ = env.step(action)

        if pair is not None:
            states, current_action = pair
            state_history = {}
            for position, observed_state in enumerate(states):
                state_history[str(position)] = observed_state
            rows.append({"timestamp": state["timestamp"], "x": state_history, "y": current_action})
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
