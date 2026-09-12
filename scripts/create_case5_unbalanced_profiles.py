"""Create phase profiles for the demand-only-unbalanced five-bus example."""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "case5"
DESTINATION = ROOT / "examples" / "case5_unbalanced"
SHARES = {
    "bus_001": {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
    "bus_002": {"a": 0.40, "b": 0.35, "c": 0.25},
    "bus_003": {"a": 0.25, "b": 0.45, "c": 0.30},
    "bus_004": {"a": 0.20, "b": 0.30, "c": 0.50},
    "bus_005": {"a": 0.45, "b": 0.25, "c": 0.30},
}


def main():
    demand = pd.read_csv(SOURCE / "demand.csv")
    output = pd.DataFrame({"timestamp": demand["timestamp"]})
    for bus, shares in SHARES.items():
        p_total = demand.get(f"P{bus}", 0.0)
        q_total = demand.get(f"Q{bus}", 0.0)
        for phase, share in shares.items():
            output[f"P{bus}_{phase}"] = p_total * share
            output[f"Q{bus}_{phase}"] = q_total * share
    output.to_csv(DESTINATION / "demand.csv", index=False)

    pv = pd.read_csv(SOURCE / "pv.csv")
    for phase in ("a", "b", "c"):
        pv[f"PV5_{phase}"] = pv["PV5"] / 3.0
    pv.to_csv(DESTINATION / "pv.csv", index=False)

    price = pd.read_csv(SOURCE / "price.csv")
    price.to_csv(DESTINATION / "price.csv", index=False)


if __name__ == "__main__":
    main()
