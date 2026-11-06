import unittest

from opf_teacher.teacher import StateActionHistory, TemporalTeacher


class StateActionHistoryTest(unittest.TestCase):
    def test_initial_padding_repeats_the_first_state_without_aliasing(self):
        history = StateActionHistory(3, pad_initial=True)
        state, action = {"value": [0]}, {"p": 10}
        x, y = history.append(state, action)
        if x != [state, state, state] or y != action:
            self.fail("Initial padding must repeat the first observation and keep the current target")
        x[0]["value"][0] = 99
        if x[1]["value"] != [0] or state["value"] != [0]:
            self.fail("Repeated observations must be independent copies")
        x, y = history.append({"value": [1]}, {"p": 20})
        if x != [{"value": [0]}, {"value": [0]}, {"value": [1]}] or y != {"p": 20}:
            self.fail("Padding must shrink as real observations accumulate")
        x, _ = history.append({"value": [2]}, {})
        if x != [{"value": [0]}, {"value": [1]}, {"value": [2]}]:
            self.fail("A full window must contain only real chronological observations")
        history.reset()
        if history.append({"value": [10]}, {})[0] != [{"value": [10]}] * 3:
            self.fail("A new episode must pad from its own first observation")

    def test_initial_padding_marks_missing_action_labels(self):
        history = StateActionHistory(3, action_history=True, pad_initial=True)
        x, y = history.append({"step": 0}, {"p": 10})
        if x != [{"step": 0}] * 3 or y != [None, None, {"p": 10}]:
            self.fail("Padding must not present the current target as a real past action")
        _, y = history.append({"step": 1}, {"p": 20})
        if y != [None, {"p": 10}, {"p": 20}]:
            self.fail("Missing action labels must disappear as history fills")

    def test_default_window_returns_the_current_pair_immediately(self):
        history = StateActionHistory()
        self.assertEqual(history.window, 1)
        self.assertEqual(history.append({"step": 0}, {"p": 10}),
                         ([{"step": 0}], {"p": 10}))
        self.assertEqual(history.append({"step": 1}, {"p": -5}),
                         ([{"step": 1}], {"p": -5}))
        history.reset()
        self.assertEqual(history.append({"step": 10}, {"p": 0}),
                         ([{"step": 10}], {"p": 0}))

    def test_window_one_copies_inputs_and_returns_independent_pairs(self):
        history = StateActionHistory()
        state = {"bus": {"v": 1.0}}
        action = {"bess": {"b1": {"p": 10}}}
        states, current_action = history.append(state, action)
        state["bus"]["v"] = 0.5
        action["bess"]["b1"]["p"] = 99
        self.assertEqual(states, [{"bus": {"v": 1.0}}])
        self.assertEqual(current_action, {"bess": {"b1": {"p": 10}}})
        history.append({"step": 1}, {})
        self.assertEqual(states, [{"bus": {"v": 1.0}}])

    def test_controller_histories_are_independent(self):
        first = StateActionHistory(window=2)
        second = StateActionHistory(window=2)
        first.append({"controller": "first", "step": 0}, {})
        self.assertIsNone(second.append({"controller": "second", "step": 0}, {}))
        first.reset()
        states, action = second.append({"controller": "second", "step": 1}, {"p": 5})
        self.assertEqual(states, [{"controller": "second", "step": 0},
                                  {"controller": "second", "step": 1}])
        self.assertEqual(action, {"p": 5})

    def test_legacy_name_preserves_its_default_window(self):
        history = TemporalTeacher()
        self.assertIsInstance(history, StateActionHistory)
        self.assertEqual(history.window, 3)
        self.assertIsNone(history.append({"step": 0}, {}))
        self.assertIsNone(history.append({"step": 1}, {}))
        self.assertEqual(history.append({"step": 2}, {"p": 5}),
                         ([{"step": 0}, {"step": 1}, {"step": 2}], {"p": 5}))
        self.assertEqual(TemporalTeacher(window=1).window, 1)

    def test_window_advances_and_uses_only_current_action(self):
        history = StateActionHistory(window=3)
        self.assertIsNone(history.append({"soc": 0.5}, {"p": 10}))
        self.assertIsNone(history.append({"soc": 0.6}, {"p": -20}))
        x, y = history.append({"soc": 0.4}, {"p": 5})
        self.assertEqual(x, [{"soc": 0.5}, {"soc": 0.6}, {"soc": 0.4}])
        self.assertEqual(y, {"p": 5})
        x, y = history.append({"soc": 0.45}, {"p": 0})
        self.assertEqual(x, [{"soc": 0.6}, {"soc": 0.4}, {"soc": 0.45}])
        self.assertEqual(y, {"p": 0})

    def test_nested_local_and_central_pairs_are_independent_copies(self):
        for state, action in (
            ({"bus": {"v": 1.0}, "device": {"soc": 0.5}}, {"p": 10}),
            ({"buses": {4: {"v": 1.0}}, "bess": {"b1": {"soc": 0.5}}},
             {"bess": {"b1": {"p": 10}}, "pv": {"pv1": {"p": 20}}}),
        ):
            history = StateActionHistory(window=2)
            history.append(state, action)
            state.clear()
            x, y = history.append({"next": [1]}, action)
            self.assertTrue(x[0])
            action.clear()
            self.assertTrue(y)
            x[1]["next"][0] = 99
            next_x, _ = history.append({"next": [2]}, {})
            self.assertEqual(next_x[0], {"next": [1]})

    def test_reset_starts_a_new_episode(self):
        history = StateActionHistory(window=2)
        history.append({"step": 0}, {})
        history.append({"step": 1}, {})
        history.reset()
        self.assertIsNone(history.append({"step": 10}, {}))
        x, _ = history.append({"step": 11}, {})
        self.assertEqual(x, [{"step": 10}, {"step": 11}])

    def test_window_one_and_invalid_sizes(self):
        self.assertEqual(StateActionHistory(window=1).append({"soc": 0.5}, {"p": 0}),
                         ([{"soc": 0.5}], {"p": 0}))
        for window in (0, -1, 1.5, True, "3", None):
            with self.subTest(window=window), self.assertRaises(ValueError):
                StateActionHistory(window=window)


if __name__ == "__main__":
    unittest.main()
