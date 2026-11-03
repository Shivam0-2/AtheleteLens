"""Unit tests for the shared AI-engine utilities."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from athletelens_ai.pose import available_landmark_data, extract_landmark_sets
from athletelens_ai.signals import optional_angle, rolling_median
from athletelens_ai.video import frame_timestamp_ms, iter_video_frames


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None


class SharedVideoTests(unittest.TestCase):
    def test_frame_iteration_yields_zero_based_indices(self):
        self.assertEqual(
            list(iter_video_frames(FakeCapture(["a", "b"]))),
            [(0, "a"), (1, "b")],
        )

    def test_frame_timestamps_preserve_rounding_and_monotonicity(self):
        self.assertEqual(frame_timestamp_ms(1, 25.0, 0), 40)
        self.assertEqual(frame_timestamp_ms(2, 25.0, 40), 80)
        # The clamp retains the existing behavior when FPS exceeds 1,000.
        self.assertEqual(frame_timestamp_ms(1, 2000.0, 0), 1)


class SharedSignalTests(unittest.TestCase):
    def test_optional_angle_preserves_missing_and_nonfinite_as_missing(self):
        self.assertIsNone(optional_angle(""))
        self.assertIsNone(optional_angle("  "))
        self.assertIsNone(optional_angle("nan"))
        self.assertIsNone(optional_angle(str(math.inf)))
        self.assertEqual(optional_angle("42.5"), 42.5)

    def test_rolling_median_keeps_gaps_and_does_not_bridge_segments(self):
        values = [1.0, 100.0, 3.0, None, 8.0, 10.0, 12.0]
        self.assertEqual(
            rolling_median(values, 3),
            [50.5, 3.0, 51.5, None, 9.0, 10.0, 11.0],
        )

    def test_rolling_median_rejects_even_window(self):
        with self.assertRaises(ValueError):
            rolling_median([1.0, 2.0], 2)


class SharedPoseTests(unittest.TestCase):
    def test_landmark_serialization_keeps_only_supplied_fields(self):
        landmark = SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9)
        self.assertEqual(
            available_landmark_data(landmark),
            {"x": 0.1, "y": 0.2, "z": 0.3, "visibility": 0.9},
        )

    def test_extract_landmark_sets_preserves_pose_and_landmark_indices(self):
        pose = [SimpleNamespace(x=0.1, y=0.2), SimpleNamespace(x=0.3, y=0.4)]
        result = SimpleNamespace(pose_landmarks=[pose], pose_world_landmarks=[])
        self.assertEqual(
            extract_landmark_sets(result, "pose_landmarks"),
            [[
                {"index": 0, "x": 0.1, "y": 0.2},
                {"index": 1, "x": 0.3, "y": 0.4},
            ]],
        )
        self.assertEqual(extract_landmark_sets(result, "pose_world_landmarks"), [])


if __name__ == "__main__":
    unittest.main()
