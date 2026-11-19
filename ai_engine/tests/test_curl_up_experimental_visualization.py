"""Small regression tests for experimental overlay event placement."""

from __future__ import annotations

import unittest

from curl_up_experimental_visualization import (
    SOURCE_VIDEO,
    _accepted_event_frames,
)


class ExperimentalVisualizationTests(unittest.TestCase):
    def test_runner_is_pinned_to_the_verified_source_filename(self):
        self.assertEqual(SOURCE_VIDEO.name, "curl up.mp4")
        self.assertNotEqual(SOURCE_VIDEO.name, "curl_up_test.mp4")

    def test_acceptance_marker_uses_count_change_frame_and_keeps_event_time(self):
        event = {
            "event_type": "accepted",
            "repetition_number": 1,
            "start_timestamp_ms": 1435,
            "end_timestamp_ms": 4204,
            "duration_seconds": 2.769,
            "reason": None,
        }
        rows = [
            {"frame_number": 125, "timestamp_ms": 4171, "repetition_count": 0},
            {"frame_number": 126, "timestamp_ms": 4204, "repetition_count": 0},
            {"frame_number": 127, "timestamp_ms": 4238, "repetition_count": 1},
        ]

        mapped = _accepted_event_frames([event], rows)

        self.assertEqual(list(mapped), [2])
        self.assertEqual(mapped[2]["end_timestamp_ms"], 4204)
        self.assertEqual(mapped[2]["accepted_on_frame_number"], 127)
        self.assertEqual(mapped[2]["accepted_on_timestamp_ms"], 4238)


if __name__ == "__main__":
    unittest.main()
