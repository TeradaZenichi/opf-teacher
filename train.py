"""Generate observed X/Y pairs with a configurable history window."""
import json
from pathlib import Path

import pandas as pd
from opendss_env import EnvironmentConfig, MicrogridEnv

from normalization import PerUnitNormalizer, TimeEncoder
from teacher import PairSelector, StateActionHistory, Teacher


# Change the run settings here.
ROOT = Path(__file__).resolve().parent
CASE_PATH = ROOT / "examples" / "case5_local"  # Input scenario.
CSV_PATH = ROOT / "results" / "train.csv"
# single_phase_active: active-power single-phase equivalent.
# single_phase_socp: single-phase P/Q model with SOCP relaxation.
# three_phase_ivr: unbalanced three-phase model with phase-specific values.
FORMULATION = "single_phase_active"
# central: all buses/devices in X and all device actions in Y.
# local: selected BESS's bus/state in X and only its action in Y.
# Both modes include encoded time and execute the complete global OPF action.
OBSERVATION_MODE = "central"
DEVICE_ID = "b1"  # BESS selected in local mode; ignored in central mode.
WINDOW = 3  # Number of observed states in each X.


def main(window=WINDOW):
    history = StateActionHistory(window=window)
    original_case = Teacher(CASE_PATH, formulation=FORMULATION).case
    selector = PairSelector(original_case, OBSERVATION_MODE, DEVICE_ID)
    normalizer = PerUnitNormalizer(original_case.base)
    time_encoder = TimeEncoder()
    env = MicrogridEnv(EnvironmentConfig(CASE_PATH, original_case.n_periods))
    rows = []
    observation, _ = env.reset()
    for step in original_case.periods:
        horizon = original_case.horizon(start=step)
        teacher = Teacher(horizon, formulation=FORMULATION)
        teacher.observe_dict(observation)
        teacher.build()
        solved = teacher.solve()
        state, action = solved.state_action()
        selected_state, selected_action = selector.select(state, action)
        normalized_state = normalizer.normalize(selected_state)
        normalized_state["time"] = time_encoder.encode(normalized_state.pop("timestamp"))
        normalized_state.pop("phase_order", None)
        normalized_action = normalizer.normalize(selected_action)
        pair = history.append(normalized_state, normalized_action)
        observation, _, terminated, truncated, _ = env.step(action)

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
    if table.empty:
        table = pd.DataFrame(columns=["timestamp"])
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    metadata = {
        "case": original_case.name, "formulation": FORMULATION,
        "observation": selector.as_dict(),
        "window": window, "periods": original_case.n_periods, "pairs": len(table),
        "phase_order": state.get("phase_order", ["a"]),
        "normalization": normalizer.as_dict(), "time_encoding": time_encoder.as_dict(),
    }
    CSV_PATH.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{len(table)} pairs saved to: {CSV_PATH} (window={window})")
    return table


if __name__ == "__main__":
    main()
