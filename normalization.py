"""Per-unit copies and time features for teacher training pairs."""
from copy import deepcopy
import math

import pandas as pd

from opf.common import Base


POWER_FIELDS = {
    "p_load_kw": "p_load_pu",
    "q_load_kvar": "q_load_pu",
    "previous_p_kw": "previous_p_pu",
    "previous_q_kvar": "previous_q_pu",
    "available_kw": "available_pu",
    "generation_kw": "generation_pu",
    "p_net_kw": "p_net_pu",
    "q_injection_kvar": "q_injection_pu",
}
PHYSICAL_POWER_FIELDS = {normalized: physical for physical, normalized in POWER_FIELDS.items()}


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


def _scale_power(value, factor):
    if isinstance(value, dict):
        return {phase: float(power) * factor for phase, power in value.items()}
    return float(value) * factor


class PerUnitNormalizer:
    """Use the case bases; phase powers also use the total network power base.

    Voltage observations are already in pu. SoC, angles, tariffs and timestamps
    retain their units. Values outside [-1, 1] are preserved.
    """

    def __init__(self, base: Base):
        self.base = Base(base.s_base_kva, base.v_base_kv)

    def normalize(self, values: dict) -> dict:
        result = {}
        for name, value in values.items():
            if name in POWER_FIELDS:
                result[POWER_FIELDS[name]] = _scale_power(value, 1.0 / self.base.s_base_kva)
            elif isinstance(value, dict):
                result[name] = self.normalize(value)
            else:
                result[name] = deepcopy(value)
        return result

    def denormalize(self, values: dict) -> dict:
        result = {}
        for name, value in values.items():
            if name in PHYSICAL_POWER_FIELDS:
                result[PHYSICAL_POWER_FIELDS[name]] = _scale_power(value, self.base.s_base_kva)
            elif isinstance(value, dict):
                result[name] = self.denormalize(value)
            else:
                result[name] = deepcopy(value)
        return result

    def as_dict(self) -> dict:
        return {
            "schema_version": 1,
            "method": "per_unit",
            "base": {
                "s_base_kva": self.base.s_base_kva,
                "v_base_kv": self.base.v_base_kv,
            },
            "phase_power_base": "total_network",
            "voltage": "already_pu",
            "clipping": False,
            "power_fields": dict(POWER_FIELDS),
        }
