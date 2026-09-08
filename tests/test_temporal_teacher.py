import unittest

from teacher import TemporalTeacher


class TemporalTeacherTest(unittest.TestCase):
    def test_window_advances_and_uses_only_current_action(self):
        teacher = TemporalTeacher(window=3)
        self.assertIsNone(teacher.append({"soc": 0.5}, {"p": 10}))
        self.assertIsNone(teacher.append({"soc": 0.6}, {"p": -20}))
        x, y = teacher.append({"soc": 0.4}, {"p": 5})
        self.assertEqual(x, [{"soc": 0.5}, {"soc": 0.6}, {"soc": 0.4}])
        self.assertEqual(y, {"p": 5})
        x, y = teacher.append({"soc": 0.45}, {"p": 0})
        self.assertEqual(x, [{"soc": 0.6}, {"soc": 0.4}, {"soc": 0.45}])
        self.assertEqual(y, {"p": 0})

    def test_nested_local_and_central_pairs_are_independent_copies(self):
        for state, action in (
            ({"bus": {"v": 1.0}, "device": {"soc": 0.5}}, {"p": 10}),
            ({"buses": {4: {"v": 1.0}}, "bess": {"b1": {"soc": 0.5}}},
             {"bess": {"b1": {"p": 10}}, "pv": {"pv1": {"p": 20}}}),
        ):
            teacher = TemporalTeacher(window=2)
            teacher.append(state, action)
            state.clear()
            x, y = teacher.append({"next": [1]}, action)
            self.assertTrue(x[0])
            action.clear()
            self.assertTrue(y)
            x[1]["next"][0] = 99
            next_x, _ = teacher.append({"next": [2]}, {})
            self.assertEqual(next_x[0], {"next": [1]})

    def test_reset_starts_a_new_episode(self):
        teacher = TemporalTeacher(window=2)
        teacher.append({"step": 0}, {})
        teacher.append({"step": 1}, {})
        teacher.reset()
        self.assertIsNone(teacher.append({"step": 10}, {}))
        x, _ = teacher.append({"step": 11}, {})
        self.assertEqual(x, [{"step": 10}, {"step": 11}])

    def test_window_one_and_invalid_sizes(self):
        self.assertEqual(TemporalTeacher(window=1).append({"soc": 0.5}, {"p": 0}),
                         ([{"soc": 0.5}], {"p": 0}))
        for window in (0, -1, 1.5, True, "3", None):
            with self.subTest(window=window), self.assertRaises(ValueError):
                TemporalTeacher(window=window)


if __name__ == "__main__":
    unittest.main()
