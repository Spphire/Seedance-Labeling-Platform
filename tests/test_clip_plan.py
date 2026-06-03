from __future__ import annotations

import math
import unittest

from app.backend.video import clip_plan, requested_seedance_duration


class ClipPlanTest(unittest.TestCase):
    def assert_plan(self, total_duration: float) -> None:
        plan = clip_plan(total_duration)
        self.assertTrue(plan)
        self.assertAlmostEqual(sum(duration for _, duration in plan), total_duration, places=6)
        for index, (start, duration) in enumerate(plan):
            self.assertGreaterEqual(duration, 4.0, (total_duration, index, plan))
            self.assertLessEqual(duration, 15.0, (total_duration, index, plan))
            if index:
                prev_start, prev_duration = plan[index - 1]
                self.assertAlmostEqual(start, prev_start + prev_duration, places=6)

    def test_required_durations(self) -> None:
        for duration in [14.2, 32.0, 32.5, 46.7, 182.3]:
            with self.subTest(duration=duration):
                self.assert_plan(duration)

    def test_request_duration_is_ceiling(self) -> None:
        for value in [4.0, 4.1, 14.2, 15.0]:
            with self.subTest(value=value):
                self.assertEqual(requested_seedance_duration(value), math.ceil(value))

    def test_short_tail_borrows_integer_seconds(self) -> None:
        for actual, expected in [
            (clip_plan(32.5), [(0.0, 15.0), (15.0, 13.0), (28.0, 4.5)]),
            (clip_plan(46.7), [(0.0, 15.0), (15.0, 15.0), (30.0, 12.0), (42.0, 4.7)]),
        ]:
            self.assertEqual(len(actual), len(expected))
            for (actual_start, actual_duration), (expected_start, expected_duration) in zip(actual, expected):
                self.assertAlmostEqual(actual_start, expected_start, places=6)
                self.assertAlmostEqual(actual_duration, expected_duration, places=6)

    def test_too_short_episode_fails_clearly(self) -> None:
        with self.assertRaises(ValueError):
            clip_plan(3.9)


if __name__ == "__main__":
    unittest.main()
