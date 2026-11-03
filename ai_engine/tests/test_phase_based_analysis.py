"""Unit tests for the experimental phase-based push-up detector."""

from __future__ import annotations

import unittest

from athletelens_ai.assessments.push_up import DetectorConfig, PhaseBasedPushUpDetector


def feed(detector: PhaseBasedPushUpDetector, values: list[float | None], step_ms: int = 100):
    for index, value in enumerate(values):
        detector.update(index + 1, index * step_ms, value)
    detector.finish(len(values) * step_ms)
    return detector


def complete_rep() -> list[float]:
    return [170, 170, 150, 150, 80, 80, 100, 100, 170, 170]


def config(**overrides) -> DetectorConfig:
    values = {
        "bottom_angle_deg": 90,
        "top_angle_deg": 160,
        "hysteresis_deg": 5,
        "confirmation_frames": 2,
        "min_repetition_duration_s": 0.2,
        "max_repetition_duration_s": 3.0,
        "max_measurement_gap_s": 0.25,
    }
    values.update(overrides)
    return DetectorConfig(**values)


class PhaseBasedDetectorTests(unittest.TestCase):
    def test_phase_detector_default_thresholds_remain_90_and_160(self):
        detector_config = DetectorConfig()
        self.assertEqual(detector_config.bottom_angle_deg, 90.0)
        self.assertEqual(detector_config.top_angle_deg, 160.0)

    def test_invalid_threshold_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bottom angle < top angle"):
            DetectorConfig(bottom_angle_deg=160, top_angle_deg=90).validate()

    def test_overlapping_hysteresis_bands_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Hysteresis bands overlap"):
            config(hysteresis_deg=40).validate()

    def test_invalid_confirmation_and_duration_configuration_is_rejected(self):
        for invalid in (
            config(confirmation_frames=0),
            config(min_repetition_duration_s=-0.1),
            config(max_repetition_duration_s=0.1, min_repetition_duration_s=0.1),
            config(max_measurement_gap_s=0),
        ):
            with self.subTest(config=invalid), self.assertRaises(ValueError):
                invalid.validate()

    def test_complete_repetition(self):
        detector = feed(PhaseBasedPushUpDetector(config()), complete_rep())
        self.assertEqual(detector.repetitions, 1)
        self.assertEqual([e["event_type"] for e in detector.events], ["completed"])

    def test_multiple_consecutive_repetitions(self):
        detector = feed(PhaseBasedPushUpDetector(config()), complete_rep() * 3)
        self.assertEqual(detector.repetitions, 3)

    def test_video_beginning_halfway_through_does_not_count_partial_rep(self):
        partial = [150, 150, 80, 80, 100, 100, 170, 170]
        detector = feed(PhaseBasedPushUpDetector(config()), partial + complete_rep())
        self.assertEqual(detector.repetitions, 1)
        reasons = [event.get("reason") for event in detector.events]
        self.assertIn("video_started_mid_repetition", reasons)

    def test_incomplete_repetition_is_rejected_at_end(self):
        detector = feed(PhaseBasedPushUpDetector(config()), [170, 170, 150, 150, 80, 80])
        self.assertEqual(detector.repetitions, 0)
        self.assertEqual(detector.events[-1]["reason"], "end_of_video_incomplete_cycle")

    def test_threshold_chatter_does_not_complete_or_duplicate_repetitions(self):
        values = [170, 170, 156, 154, 156, 154, 170, 170]
        detector = feed(PhaseBasedPushUpDetector(config()), values + complete_rep())
        self.assertEqual(detector.repetitions, 1)
        self.assertEqual(sum(e["event_type"] == "completed" for e in detector.events), 1)

    def test_missing_frame_during_each_confirmed_phase_does_not_invent_transition(self):
        phases = [
            [170, 170],  # TOP
            [150, 150],  # DESCENDING
            [80, 80],    # BOTTOM
            [100, 100],  # ASCENDING
        ]
        for phase_index in range(len(phases)):
            with self.subTest(phase=phase_index):
                values: list[float | None] = []
                for index, phase in enumerate(phases):
                    values.extend(phase)
                    if index == phase_index:
                        values.append(None)
                values.extend([170, 170])
                detector = feed(PhaseBasedPushUpDetector(config()), values)
                self.assertEqual(detector.repetitions, 1)
                self.assertFalse(any(e["event_type"] == "rejected" for e in detector.events))

    def test_too_short_repetition_is_rejected(self):
        detector = feed(
            PhaseBasedPushUpDetector(
                config(min_repetition_duration_s=0.8, confirmation_frames=1)
            ),
            [170, 150, 80, 100, 170],
            step_ms=100,
        )
        self.assertEqual(detector.repetitions, 0)
        self.assertIn("too_short", [e.get("reason") for e in detector.events])

    def test_excessively_long_repetition_is_rejected(self):
        values = [170, 170, 150, 150] + [120] * 15 + [80, 80, 100, 100, 170, 170]
        detector = feed(
            PhaseBasedPushUpDetector(
                config(max_repetition_duration_s=1.0, max_measurement_gap_s=0.25)
            ),
            values,
        )
        self.assertEqual(detector.repetitions, 0)
        self.assertIn("excessively_long", [e.get("reason") for e in detector.events])

    def test_long_measurement_gap_invalidates_incomplete_cycle(self):
        values: list[float | None] = [170, 170, 150, 150, 80, 80]
        values.extend([None] * 4)
        values.extend([100, 100, 170, 170])
        detector = feed(PhaseBasedPushUpDetector(config()), values)
        self.assertEqual(detector.repetitions, 0)
        self.assertIn("long_measurement_gap", [e.get("reason") for e in detector.events])


if __name__ == "__main__":
    unittest.main()
