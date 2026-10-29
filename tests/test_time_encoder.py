from copy import deepcopy
import json
import math
import unittest

import pandas as pd

from normalization import PerUnitNormalizer, TimeEncoder
from opf.common import Base
from teacher import StateActionHistory


class TimeEncoderTest(unittest.TestCase):
    def setUp(self):
        self.encoder = TimeEncoder()

    def test_hour_features_follow_the_daily_cycle(self):
        cases = ((0, (0.0, 1.0)), (6, (1.0, 0.0)), (12, (0.0, -1.0)), (18, (-1.0, 0.0)))
        for hour, expected in cases:
            with self.subTest(hour=hour):
                features = self.encoder.encode(f"2026-01-01T{hour:02d}:00:00")
                self.assertAlmostEqual(features["hour_sin"], expected[0])
                self.assertAlmostEqual(features["hour_cos"], expected[1])

    def test_subhour_intervals_keep_minutes_seconds_and_fractional_seconds(self):
        features = self.encoder.encode("2026-01-01T00:30:30.500000")
        angle = 2.0 * math.pi * 1830.5 / 86400.0
        self.assertAlmostEqual(features["hour_sin"], math.sin(angle))
        self.assertAlmostEqual(features["hour_cos"], math.cos(angle))

    def test_midnight_keeps_neighboring_times_close(self):
        before = self.encoder.encode("2026-01-01T23:45:00")
        after = self.encoder.encode("2026-01-02T00:15:00")
        distance = math.hypot(
            before["hour_sin"] - after["hour_sin"],
            before["hour_cos"] - after["hour_cos"],
        )
        self.assertLess(distance, 0.14)
        self.assertEqual(self.encoder.encode("2026-01-01T00:00:00"),
                         self.encoder.encode("2026-01-02T00:00:00"))

    def test_time_uses_the_timestamp_clock_without_implicit_timezone_conversion(self):
        local = self.encoder.encode("2026-01-01T06:00:00-03:00")
        same_clock = self.encoder.encode("2026-01-01T06:00:00Z")
        same_instant = self.encoder.encode("2026-01-01T09:00:00Z")
        self.assertEqual(local, same_clock)
        self.assertNotEqual(local, same_instant)
        self.assertEqual(local, self.encoder.encode(pd.Timestamp("2026-01-01T06:00:00")))

    def test_empty_or_invalid_timestamps_are_rejected(self):
        for timestamp in (None, pd.NaT, "NaT", "invalid"):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.encoder.encode(timestamp)

    def test_encoding_is_recorded_in_serializable_metadata(self):
        metadata = json.loads(json.dumps(self.encoder.as_dict()))
        self.assertEqual(metadata["method"], "sine_cosine")
        self.assertEqual(metadata["period_hours"], 24)
        self.assertEqual(metadata["features"], ["hour_sin", "hour_cos"])

    def test_each_window_position_uses_its_own_observed_time(self):
        normalizer = PerUnitNormalizer(Base(1000.0, 11.0))
        history = StateActionHistory(window=3)
        physical = {
            "timestamp": "2026-01-01T00:00:00",
            "bess": {"b1": {"soc_before_frac": 0.5, "previous_p_kw": -20.0}},
        }
        original = deepcopy(physical)
        for hour in (0, 6, 12):
            observed = deepcopy(physical)
            observed["timestamp"] = f"2026-01-01T{hour:02d}:00:00"
            features = normalizer.normalize(observed)
            features["time"] = self.encoder.encode(features.pop("timestamp"))
            pair = history.append(features, {"step_hour": hour})
        states, action = pair
        self.assertEqual(physical, original)
        for state, hour in zip(states, (0, 6, 12)):
            self.assertNotIn("timestamp", state)
            self.assertEqual(state["time"], self.encoder.encode(f"2026-01-01T{hour:02d}:00:00"))
            self.assertEqual(state["bess"]["b1"]["soc_before_frac"], 0.5)
            self.assertEqual(state["bess"]["b1"]["previous_p_pu"], -0.02)
        self.assertEqual(action, {"step_hour": 12})


if __name__ == "__main__":
    unittest.main()
