"""Synthetic tests for configurable curl-up measurement and phase logic."""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from athletelens_ai.assessments.base import AssessmentMeasurement
from athletelens_ai.assessments.curl_up import (
    CurlUpAssessment,
    CurlUpConfig,
    CurlUpPhaseDetector,
)
from athletelens_ai.pipeline import PoseObservation
from athletelens_ai.video import VideoMetadata


def config(**overrides) -> CurlUpConfig:
    values = {
        "image_aspect_ratio": 1.0,
        "extended_angle_deg": 150.0,
        "flexed_angle_deg": 120.0,
        "hysteresis_deg": 5.0,
        "confirmation_frames": 2,
        "min_repetition_duration_s": 0.2,
        "max_repetition_duration_s": 3.0,
        "max_measurement_gap_s": 0.25,
        "min_visibility": 0.5,
        "starting_position": "extended",
        "selected_side": "auto",
        "test_duration_s": None,
    }
    values.update(overrides)
    return CurlUpConfig(**values)


def sample(
    frame: int,
    timestamp_ms: int,
    angle: float | None,
    *,
    pose_detected: bool = True,
    missing_landmarks: str = "",
) -> AssessmentMeasurement:
    return AssessmentMeasurement(
        frame_number=frame,
        timestamp_ms=timestamp_ms,
        values={
            "included_angle_deg": angle,
            "selected_side": "left" if angle is not None else "",
        },
        quality={
            "pose_detected": pose_detected,
            "measurement_valid": angle is not None,
            "visibility_score": 0.9 if angle is not None else None,
            "missing_landmarks": missing_landmarks,
        },
    )


def analyze_angles(
    angles: list[float | None],
    *,
    timestamps: list[int] | None = None,
    test_config: CurlUpConfig | None = None,
):
    timestamps = timestamps or [index * 100 for index in range(len(angles))]
    inputs = [sample(index, ts, angle) for index, (ts, angle) in enumerate(zip(timestamps, angles))]
    return CurlUpAssessment(test_config or config()).analyze(
        inputs, VideoMetadata(fps=10.0, width=640, height=480)
    )


def pose_observation(
    frame: int,
    timestamp_ms: int,
    *,
    missing: tuple[int, ...] = (),
    visibility: float = 0.9,
) -> PoseObservation:
    landmarks = [None] * 33
    for index, x, y in (
        (11, 0.0, 0.0),  # left shoulder
        (23, 1.0, 0.0),  # left hip
        (25, 2.0, 0.0),  # left knee; straight angle is 180 degrees
    ):
        if index not in missing:
            landmarks[index] = SimpleNamespace(x=x, y=y, visibility=visibility)
    return PoseObservation(
        frame_number=frame,
        timestamp_ms=timestamp_ms,
        pose_landmarks=[landmarks],
        pose_world_landmarks=[],
        pose_landmark_data=[],
        pose_world_landmark_data=[],
        result=SimpleNamespace(pose_landmarks=[landmarks]),
    )


class CurlUpMeasurementTests(unittest.TestCase):
    def test_measurement_uses_aspect_corrected_shoulder_hip_knee_angle(self):
        assessment = CurlUpAssessment(config())
        measured = assessment.measure_frame(pose_observation(4, 160))

        self.assertAlmostEqual(measured.values["included_angle_deg"], 180.0)
        self.assertAlmostEqual(measured.values["curl_flexion_deg"], 0.0)
        self.assertEqual(measured.values["selected_side"], "left")
        self.assertTrue(measured.quality["measurement_valid"])
        self.assertAlmostEqual(measured.quality["visibility_score"], 0.9)

    def test_missing_knee_landmark_is_reported_and_not_measured(self):
        assessment = CurlUpAssessment(config(selected_side="left"))
        measured = assessment.measure_frame(
            pose_observation(2, 80, missing=(25,))
        )

        self.assertIsNone(measured.values["included_angle_deg"])
        self.assertFalse(measured.quality["measurement_valid"])
        self.assertIn("left_knee", measured.quality["missing_landmarks"])

    def test_missing_visibility_is_not_assumed_to_be_good_quality(self):
        assessment = CurlUpAssessment(config(selected_side="left"))
        landmarks = pose_observation(0, 0).pose_landmarks[0]
        landmarks[11] = SimpleNamespace(x=0.0, y=0.0)
        observation = PoseObservation(
            frame_number=0,
            timestamp_ms=0,
            pose_landmarks=[landmarks],
            pose_world_landmarks=[],
            pose_landmark_data=[],
            pose_world_landmark_data=[],
            result=SimpleNamespace(pose_landmarks=[landmarks]),
        )

        measured = assessment.measure_frame(observation)

        self.assertIsNone(measured.values["included_angle_deg"])
        self.assertIn("left_shoulder_visibility", measured.quality["missing_landmarks"])


class CurlUpPhaseDetectorTests(unittest.TestCase):
    def test_complete_configured_cycle_is_accepted_with_timestamps(self):
        result = analyze_angles(
            [160, 160, 140, 140, 110, 110, 130, 130, 160, 160]
        )

        self.assertEqual(result.summary["detected_repetitions"], 1)
        accepted = result.summary["accepted_repetitions"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["start_timestamp_ms"], 200)
        self.assertEqual(accepted[0]["end_timestamp_ms"], 800)
        self.assertEqual(accepted[0]["duration_seconds"], 0.6)
        self.assertEqual(result.summary["protocol_status"], "experimental_protocol_not_defined_in_srs")
        serialized = json.loads(json.dumps(asdict(result)))
        self.assertEqual(serialized["events"][0]["event_type"], "accepted")
        self.assertEqual(serialized["summary"]["measurements"][2]["timestamp_ms"], 200)

    def test_incomplete_flexed_cycle_is_rejected_at_end(self):
        result = analyze_angles([160, 160, 140, 140, 110, 110])

        self.assertEqual(result.summary["detected_repetitions"], 0)
        self.assertIn(
            "end_of_test_incomplete_cycle",
            [event["reason"] for event in result.events],
        )

    def test_missing_landmark_measurement_does_not_fake_a_flexion_transition(self):
        inputs = [
            sample(0, 0, 160),
            sample(1, 100, 160),
            sample(2, 200, 140),
            sample(3, 300, 140),
            sample(4, 400, None, missing_landmarks="left_knee"),
            sample(5, 500, 140),
            sample(6, 600, 160),
            sample(7, 700, 160),
        ]
        result = CurlUpAssessment(config()).analyze(
            inputs, VideoMetadata(fps=10.0, width=640, height=480)
        )

        self.assertEqual(result.summary["detected_repetitions"], 0)
        self.assertEqual(result.summary["frames_with_valid_measurements"], 7)
        self.assertEqual(result.summary["measurements"][4]["missing_landmarks"], "left_knee")

    def test_threshold_chatter_does_not_accept_an_incomplete_cycle(self):
        result = analyze_angles(
            [160, 160, 146, 144, 146, 144, 146, 160, 160]
        )

        self.assertEqual(result.summary["detected_repetitions"], 0)
        self.assertFalse(any(event["event_type"] == "accepted" for event in result.events))

    def test_long_measurement_gap_rejects_active_cycle(self):
        result = analyze_angles(
            [160, 160, 140, 140, None, 110, 110, 130, 130, 160, 160],
            timestamps=[0, 100, 200, 300, 700, 800, 900, 1000, 1100, 1200, 1300],
        )

        self.assertEqual(result.summary["detected_repetitions"], 0)
        self.assertIn("long_measurement_gap", [event["reason"] for event in result.events])

    def test_configured_duration_limits_analysis_window(self):
        duration_config = config(test_duration_s=0.55)
        result = analyze_angles(
            [160, 160, 140, 140, 110, 110, 130, 130, 160, 160],
            test_config=duration_config,
        )

        self.assertEqual(result.summary["total_frames_analyzed"], 6)
        self.assertEqual(result.summary["detected_repetitions"], 0)
        self.assertEqual(result.events[-1]["reason"], "end_of_test_incomplete_cycle")

    def test_invalid_threshold_and_hysteresis_configurations_are_rejected(self):
        for invalid in (
            config(flexed_angle_deg=150.0, extended_angle_deg=150.0),
            config(hysteresis_deg=16.0),
            config(max_measurement_gap_s=0.0),
            config(test_duration_s=0.0),
        ):
            with self.subTest(config=invalid), self.assertRaises(ValueError):
                CurlUpPhaseDetector(invalid)


if __name__ == "__main__":
    unittest.main()
