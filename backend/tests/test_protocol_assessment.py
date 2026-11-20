"""Shared protocol-runner tests with mocked observations and video I/O."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from athletelens_ai.assessments.base import AssessmentMeasurement, AssessmentResult
from athletelens_ai.assessments.broad_jump import BroadJumpAssessment, BroadJumpConfig
from athletelens_ai.assessments.curl_up import CurlUpAssessment, CurlUpConfig
from athletelens_ai.assessments.vertical_jump import VerticalJumpAssessment
from athletelens_ai.pipeline import PoseObservation
from athletelens_ai.video import VideoMetadata

from app.services.errors import InvalidVideoError, MissingPoseModelError
from app.services.protocol_assessment import assess_protocol_video


class FakePipeline:
    instances: list[FakePipeline] = []

    def __init__(self, video_path: Path, model_path: Path):
        self.video_path = video_path
        self.model_path = model_path
        self.metadata = VideoMetadata(30.0, 640, 480)
        self.closed = False
        self.observations = [
            PoseObservation(i, i * 33, [], [], [], [], SimpleNamespace())
            for i in range(3)
        ]
        self.instances.append(self)

    def __enter__(self):
        return self

    def __iter__(self):
        return iter(self.observations)

    def __exit__(self, *_args):
        self.closed = True


class ProtocolAssessmentServiceTests(unittest.TestCase):
    def setUp(self):
        FakePipeline.instances.clear()

    def test_shared_runner_integrates_all_protocol_assessments(self):
        factories = {
            "curl_up": lambda metadata: CurlUpAssessment(CurlUpConfig(
                image_aspect_ratio=metadata.width / metadata.height,
                extended_angle_deg=150, flexed_angle_deg=100, hysteresis_deg=5,
                confirmation_frames=2, min_repetition_duration_s=.25,
                max_repetition_duration_s=3, max_measurement_gap_s=.25,
                min_visibility=.5, starting_position="extended",
                selected_side="auto", test_duration_s=None,
            )),
            "vertical_jump": lambda _metadata: VerticalJumpAssessment(),
            "standing_broad_jump": lambda _metadata: BroadJumpAssessment(BroadJumpConfig()),
        }
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.mp4"
            model = Path(directory) / "pose.task"
            video.write_bytes(b"video")
            model.write_bytes(b"model")

            for exercise, factory in factories.items():
                with self.subTest(exercise=exercise):
                    response = assess_protocol_video(
                        video, model, "sample.mp4", factory,
                        pipeline_factory=FakePipeline,
                    )
                    self.assertEqual(response.exercise_type, exercise)
                    self.assertEqual(response.processing_status, "completed")
                    self.assertEqual(response.total_frames_processed, 3)
                    self.assertEqual(response.frames_with_detected_poses, 0)
                    self.assertEqual(response.video.width, 640)
                    self.assertFalse(response.annotated_video_available)
                    self.assertIsNone(response.annotated_video_path)
                    json.loads(response.model_dump_json())

        self.assertTrue(all(instance.closed for instance in FakePipeline.instances))

    def test_missing_files_are_rejected_before_pipeline_creation(self):
        factory = lambda _metadata: BroadJumpAssessment()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(InvalidVideoError):
                assess_protocol_video(root / "missing.mp4", root / "model.task", "x.mp4", factory,
                                      pipeline_factory=FakePipeline)
            video = root / "sample.mp4"
            video.write_bytes(b"video")
            with self.assertRaises(MissingPoseModelError):
                assess_protocol_video(video, root / "missing.task", "x.mp4", factory,
                                      pipeline_factory=FakePipeline)
        self.assertEqual(FakePipeline.instances, [])

    def test_service_preserves_accepted_and_rejected_events_in_structured_result(self):
        accepted = {
            "event_type": "accepted", "attempt_status": "accepted", "attempt_number": 1,
            "horizontal_displacement_px": 356.0, "horizontal_displacement_units": "px",
            "physical_distance_available": False, "distance_m": None, "distance_cm": None,
            "calibration_status": "unavailable", "reason": None,
        }
        rejected = {"event_type": "rejected", "reason": "incomplete_attempt"}

        class SyntheticAssessment:
            def measure_frame(self, observation):
                return AssessmentMeasurement(
                    observation.frame_number, observation.timestamp_ms,
                    {"heel_center_x_px": 10.0},
                    {"measurement_valid": True, "visibility_score": .8},
                )

            def analyze(self, measurements, _video):
                return AssessmentResult(
                    "standing_broad_jump",
                    {"attempt_count": 1, "accepted_attempts": [accepted],
                     "rejected_attempts": [rejected], "physical_distance_available": False,
                     "measurement_coverage_percentage": 100},
                    events=(accepted, rejected), configuration={},
                )

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.mp4"
            model = Path(directory) / "pose.task"
            video.write_bytes(b"video")
            model.write_bytes(b"model")
            response = assess_protocol_video(
                video, model, "sample.mp4", lambda _metadata: SyntheticAssessment(),
                pipeline_factory=FakePipeline,
            )

        self.assertEqual(response.accepted_attempt_count, 1)
        self.assertEqual(response.accepted_attempts[0]["horizontal_displacement_units"], "px")
        self.assertIsNone(response.accepted_attempts[0]["distance_cm"])
        self.assertEqual(response.rejected_attempts[0]["reason"], "incomplete_attempt")
        self.assertFalse(response.physical_distance_available)
        json.loads(response.model_dump_json())

    def test_pipeline_measurement_failure_still_closes_pipeline(self):
        class BrokenAssessment:
            def measure_frame(self, _observation):
                raise RuntimeError("synthetic")

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.mp4"
            model = Path(directory) / "pose.task"
            video.write_bytes(b"video")
            model.write_bytes(b"model")
            with self.assertRaises(Exception) as raised:
                assess_protocol_video(video, model, "sample.mp4", lambda _metadata: BrokenAssessment(),
                                      pipeline_factory=FakePipeline)
        from app.services.errors import AssessmentProcessingError

        self.assertIsInstance(raised.exception, AssessmentProcessingError)
        self.assertTrue(FakePipeline.instances[-1].closed)


if __name__ == "__main__":
    unittest.main()
