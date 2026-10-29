"""Per-unit copies and time features for teacher training pairs."""
import math

import pandas as pd

from opf.common import Base
from pu_normalizer import PUNormalizer


class TimeEncoder:
    """Encode the timestamp's local time on a 24-hour circle."""

    def encode(self, timestamp) -> dict:
        time = pd.Timestamp(timestamp)
        if pd.isna(time):
            raise ValueError("timestamp must identify a valid time")
        seconds = (
            time.hour * 3600 + time.minute * 60 + time.second
            + time.microsecond / 1e6 + time.nanosecond / 1e9
        )
        angle = 2.0 * math.pi * seconds / 86400.0
        return {"hour_sin": math.sin(angle), "hour_cos": math.cos(angle)}

    def as_dict(self) -> dict:
        return {
            "method": "sine_cosine",
            "period_hours": 24,
            "clock": "timestamp_local",
            "features": ["hour_sin", "hour_cos"],
        }


class PerUnitNormalizer(PUNormalizer):
    """Compatibility constructor for callers that pass an OPF Base object."""

    def __init__(self, base: Base):
        super().__init__(base.s_base_kva, base.v_base_kv)
