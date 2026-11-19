"""Tests for the shared one-pass video-to-pose pipeline."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from athletelens_ai.pipeline import PoseVideoPipeline
from athletelens_ai.video import VideoMetadata


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
        self.landmarker = object()

    def __enter__(self):
        return self.landmarker

    def __exit__(self, *_exc):
        self.closed = True


def make_pipeline(frames, results, **overrides):
    capture = FakeCapture(frames)
    landmarker_context = FakeLandmarkerContext()
    dependencies = (object(), object(), object())
    defaults = {
        "dependency_loader": lambda: dependencies,
        "video_opener": lambda *_args: (capture, VideoMetadata(25.0, 640, 480)),
        "landmarker_factory": lambda *_args: landmarker_context,
        "frame_processor": lambda _frame, _timestamp, *_args: next(results),
    }
    defaults.update(overrides)
    pipeline = PoseVideoPipeline("sample.mp4", "pose.task", **defaults)
    return pipeline, capture, landmarker_context


class PoseVideoPipelineTests(unittest.TestCase):
    def test_frames_have_zero_based_numbers_and_monotonic_timestamps(self):
        results = iter(
            [
                SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[]),
                SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[]),
                SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[]),
            ]
        )
        pipeline, capture, landmarker = make_pipeline(["a", "b", "c"], results)
        with pipeline:
            observations = list(pipeline)

        self.assertEqual([o.frame_number for o in observations], [0, 1, 2])
        self.assertEqual([o.timestamp_ms for o in observations], [0, 40, 80])
        self.assertTrue(capture.released)
        self.assertTrue(landmarker.closed)

    def test_missing_pose_is_an_empty_observation_and_missing_fields_stay_missing(self):
        result = SimpleNamespace(
            pose_landmarks=None,
            pose_world_landmarks=None,
        )
        pipeline, *_ = make_pipeline(["frame"], iter([result]))
        with pipeline:
            observation = next(iter(pipeline))

        self.assertFalse(observation.pose_detected)
        self.assertEqual(observation.pose_landmarks, [])
        self.assertEqual(observation.pose_landmark_data, [])
        self.assertEqual(observation.pose_world_landmark_data, [])
        self.assertIs(observation.result, result)

    def test_detected_landmarks_preserve_only_fields_supplied(self):
        landmark = SimpleNamespace(x=0.25, y=0.5, z=-0.1, visibility=0.8)
        result = SimpleNamespace(
            pose_landmarks=[[landmark]], pose_world_landmarks=[]
        )
        pipeline, *_ = make_pipeline(["frame"], iter([result]))
        with pipeline:
            observation = next(iter(pipeline))

        self.assertTrue(observation.pose_detected)
        self.assertEqual(
            observation.pose_landmark_data,
            [[{"index": 0, "x": 0.25, "y": 0.5, "z": -0.1, "visibility": 0.8}]],
        )

    def test_resources_are_released_when_frame_processing_fails(self):
        def fail(*_args):
            raise RuntimeError("synthetic processing error")

        pipeline, capture, landmarker = make_pipeline(
            ["frame"], iter(()), frame_processor=fail
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic processing error"):
            with pipeline:
                list(pipeline)
        self.assertTrue(capture.released)
        self.assertTrue(landmarker.closed)

    def test_capture_is_released_if_landmarker_creation_fails(self):
        def fail(*_args):
            raise RuntimeError("model initialization failed")

        pipeline, capture, _landmarker = make_pipeline(
            [], iter(()), landmarker_factory=fail
        )
        with self.assertRaisesRegex(RuntimeError, "model initialization failed"):
            pipeline.__enter__()
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
