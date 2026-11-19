"""Integration tests for the pose-pipeline and assessment protocol boundary."""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from typing import Sequence

from athletelens_ai.assessments.base import (
    AssessmentMeasurement,
    AssessmentProtocol,
    AssessmentResult,
)
from athletelens_ai.pipeline import PoseObservation, PoseVideoPipeline
from athletelens_ai.video import VideoMetadata


class SyntheticAssessment:
    """A deterministic test assessment driven by a landmark x coordinate."""

    def measure_frame(self, observation: PoseObservation) -> AssessmentMeasurement:
        has_pose = observation.pose_detected
        value = (
            float(observation.pose_landmarks[0][0].x)
            if has_pose and observation.pose_landmarks[0]
            else None
        )
        return AssessmentMeasurement(
            frame_number=observation.frame_number,
            timestamp_ms=observation.timestamp_ms,
            values={"synthetic_value": value},
            quality={"pose_detected": has_pose},
        )

    def analyze(
        self,
        measurements: Sequence[AssessmentMeasurement],
        video: VideoMetadata,
    ) -> AssessmentResult:
        observed = [
            float(item.values["synthetic_value"])
            for item in measurements
            if item.values["synthetic_value"] is not None
        ]
        missing = sum(
            not bool(item.quality["pose_detected"]) for item in measurements
        )
        return AssessmentResult(
            assessment_type="synthetic_landmark_signal",
            summary={
                "frames": len(measurements),
                "valid_measurements": len(observed),
                "missing_pose_frames": missing,
                "mean_value": sum(observed) / len(observed) if observed else None,
                "video_fps": video.fps,
            },
            events=(
                {
                    "frame_number": measurements[-1].frame_number,
                    "timestamp_ms": measurements[-1].timestamp_ms,
                    "value": measurements[-1].values["synthetic_value"],
                },
            ) if measurements else (),
            configuration={"synthetic_scale": 1.0},
        )


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.released = False

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


class FakeLandmarkerContext:
    def __init__(self):
        self.closed = False

    def __enter__(self):
        return object()

    def __exit__(self, *_exc):
        self.closed = True


class AssessmentProtocolIntegrationTests(unittest.TestCase):
    def _pipeline(self):
        frames = ["pose-low", "no-pose", "pose-high"]
        pose_results = iter(
            [
                SimpleNamespace(
                    pose_landmarks=[[SimpleNamespace(x=0.2)]],
                    pose_world_landmarks=[],
                ),
                SimpleNamespace(pose_landmarks=None, pose_world_landmarks=None),
                SimpleNamespace(
                    pose_landmarks=[[SimpleNamespace(x=0.8)]],
                    pose_world_landmarks=[],
                ),
            ]
        )
        capture = FakeCapture(frames)
        landmarker_context = FakeLandmarkerContext()
        metadata = VideoMetadata(fps=25.0, width=640, height=480)
        pipeline = PoseVideoPipeline(
            "synthetic.mp4",
            "synthetic.task",
            dependency_loader=lambda: (object(), object(), object()),
            video_opener=lambda *_args: (capture, metadata),
            landmarker_factory=lambda *_args: landmarker_context,
            frame_processor=lambda _frame, *_args: next(pose_results),
        )
        return pipeline, capture, landmarker_context

    def test_synthetic_assessment_handles_missing_pose_and_serializes_result(self):
        assessment: AssessmentProtocol = SyntheticAssessment()
        pipeline, capture, landmarker_context = self._pipeline()

        with pipeline:
            measurements = [
                assessment.measure_frame(observation)
                for observation in pipeline
            ]
            result = assessment.analyze(measurements, pipeline.metadata)

        self.assertEqual(
            [item.timestamp_ms for item in measurements], [0, 40, 80]
        )
        self.assertEqual(
            [item.frame_number for item in measurements], [0, 1, 2]
        )
        self.assertEqual(
            [item.values["synthetic_value"] for item in measurements],
            [0.2, None, 0.8],
        )
        self.assertEqual(
            [item.quality["pose_detected"] for item in measurements],
            [True, False, True],
        )
        self.assertEqual(
            result.summary,
            {
                "frames": 3,
                "valid_measurements": 2,
                "missing_pose_frames": 1,
                "mean_value": 0.5,
                "video_fps": 25.0,
            },
        )
        self.assertEqual(
            list(result.events),
            [{"frame_number": 2, "timestamp_ms": 80, "value": 0.8}],
        )
        serialized = json.loads(json.dumps(asdict(result)))
        self.assertEqual(serialized["assessment_type"], "synthetic_landmark_signal")
        self.assertEqual(serialized["summary"]["missing_pose_frames"], 1)
        self.assertEqual(serialized["events"][0]["timestamp_ms"], 80)
        self.assertTrue(capture.released)
        self.assertTrue(landmarker_context.closed)

    def test_measurement_preserves_timestamp_for_missing_pose(self):
        assessment = SyntheticAssessment()
        observation = PoseObservation(
            frame_number=7,
            timestamp_ms=280,
            pose_landmarks=[],
            pose_world_landmarks=[],
            pose_landmark_data=[],
            pose_world_landmark_data=[],
            result=SimpleNamespace(pose_landmarks=None),
        )

        measurement = assessment.measure_frame(observation)

        self.assertEqual(measurement.frame_number, 7)
        self.assertEqual(measurement.timestamp_ms, 280)
        self.assertIsNone(measurement.values["synthetic_value"])
        self.assertFalse(measurement.quality["pose_detected"])


if __name__ == "__main__":
    unittest.main()
