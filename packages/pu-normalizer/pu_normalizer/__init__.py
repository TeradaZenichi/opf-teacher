"""Shared per-unit conversions with explicit field names and fixed bases."""
from copy import deepcopy
import math


POWER_FIELDS = {
    "p_load_kw": "p_load_pu",
    "q_load_kvar": "q_load_pu",
    "previous_p_kw": "previous_p_pu",
    "previous_q_kvar": "previous_q_pu",
    "available_kw": "available_pu",
    "generation_kw": "generation_pu",
    "p_net_kw": "p_net_pu",
    "q_injection_kvar": "q_injection_pu",
    "grid_kw": "grid_p_pu",
    "grid_kvar": "grid_q_pu",
    "grid_import_kw": "grid_import_pu",
    "grid_export_kw": "grid_export_pu",
    "p_net_total_kw": "p_net_total_pu",
    "generation_total_kw": "generation_total_pu",
    "p_injection_total_kw": "p_injection_total_pu",
    "q_injection_total_kvar": "q_injection_total_pu",
    "curtailment_kw": "curtailment_pu",
    "inverter_loss_kw": "inverter_loss_pu",
    "grid_consumption_kw": "grid_consumption_pu",
}
PHYSICAL_POWER_FIELDS = {normalized: physical for physical, normalized in POWER_FIELDS.items()}


def _scale_power(value, factor):
    if isinstance(value, dict):
        return {phase: _scale_power(power, factor) for phase, power in value.items()}
    power = float(value)
    if not math.isfinite(power):
        raise ValueError("Power values must be finite")
    return power * factor


class PUNormalizer:
    """Use one total power base for aggregate and phase powers.

    Voltages already in pu, SoC fractions, angles, tariffs and timestamps are
    preserved. The voltage base records the scenario convention; it is never
    applied a second time to an existing pu voltage. Values are not clipped.
    """

    def __init__(self, s_base_kva, v_base_kv):
        self.s_base_kva = float(s_base_kva)
        self.v_base_kv = float(v_base_kv)
        for name, value in (("s_base_kva", self.s_base_kva), ("v_base_kv", self.v_base_kv)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    def normalize(self, values):
        result = {}
        for name, value in values.items():
            if name in POWER_FIELDS:
                result[POWER_FIELDS[name]] = _scale_power(value, 1.0 / self.s_base_kva)
            elif isinstance(value, dict):
                result[name] = self.normalize(value)
            else:
                result[name] = deepcopy(value)
        return result

    def denormalize(self, values):
        result = {}
        for name, value in values.items():
            if name in PHYSICAL_POWER_FIELDS:
                result[PHYSICAL_POWER_FIELDS[name]] = _scale_power(value, self.s_base_kva)
            elif isinstance(value, dict):
                result[name] = self.denormalize(value)
            else:
                result[name] = deepcopy(value)
        return result

    def require_pu(self, values):
        for name, value in values.items():
            if isinstance(name, str) and name.endswith(("_kw", "_kvar")):
                raise ValueError(f"Expected per-unit fields; received physical field {name!r}")
            if isinstance(value, dict):
                self.require_pu(value)

    def check_base(self, base):
        expected = {"s_base_kva": self.s_base_kva, "v_base_kv": self.v_base_kv}
        for name, value in expected.items():
            if name not in base or not math.isclose(float(base[name]), value, rel_tol=1e-12):
                raise ValueError(f"Observation {name} must match the case base")

    def as_dict(self):
        return {
            "schema_version": 1,
            "method": "per_unit",
            "base": {"s_base_kva": self.s_base_kva, "v_base_kv": self.v_base_kv},
            "phase_power_base": "total_network",
            "voltage": "already_pu",
            "clipping": False,
            "power_fields": dict(POWER_FIELDS),
        }


__all__ = ["PUNormalizer"]
