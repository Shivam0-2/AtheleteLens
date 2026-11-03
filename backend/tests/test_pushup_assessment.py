"""Mocked service and endpoint tests for the video-upload prototype."""

from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.v1 import router as api_v1
from app.core.config import Settings
from app.schemas.assessment import (
    DetectorConfiguration,
    PushUpAssessmentResponse,
    VideoMetadata,
)
from app.services import pushup_assessment as service


class FakeCapture:
    def __init__(self, frames: int):
        self.frames = iter(range(frames))
        self.released = False

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


class FakeLandmarker:
    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.closed = True


def response_fixture(filename: str = "sample.mp4") -> PushUpAssessmentResponse:
    return PushUpAssessmentResponse(
        input_filename=filename,
        video=VideoMetadata(fps=25, width=640, height=480),
        total_frames_processed=1,
        frames_with_detected_poses=1,
        frames_with_valid_elbow_angle=1,
        pose_detection_percentage=100,
        detected_repetitions=0,
        average_elbow_angle_degrees=120,
        minimum_elbow_angle_degrees=120,
        maximum_elbow_angle_degrees=120,
        detector_configuration=DetectorConfiguration(
            bottom_angle_deg=90,
            top_angle_deg=160,
            hysteresis_deg=5,
            confirmation_frames=2,
            min_repetition_duration_s=0.25,
            max_repetition_duration_s=3,
            max_measurement_gap_s=0.25,
            smoothing_window_frames=7,
        ),
        accepted_repetition_timestamps=[],
        rejected_events=[],
    )


class PushUpAssessmentServiceTests(unittest.TestCase):
    def _run_with_mocks(self, video_path: Path, model_path: Path):
        # Eight samples per phase leave enough valid support for the shared
        # centered seven-frame median while keeping the cycle under 3 seconds.
        angles = [170.0] * 8 + [145.0] * 8 + [80.0] * 8 + [120.0] * 8 + [170.0] * 8
        capture = FakeCapture(len(angles))
        landmarker = FakeLandmarker()
        result_by_frame = lambda frame: SimpleNamespace(pose_landmarks=[[frame]])
        angle_for_pose = lambda pose, _minimum: (angles[pose[0]], "left", None, None, {})
        with (
            patch.object(service, "load_pose_dependencies", return_value=(object(), object(), object())),
            patch.object(
                service,
                "open_video",
                return_value=(capture, service.EngineVideoMetadata(25.0, 640, 480)),
            ),
            patch.object(service, "create_video_landmarker", return_value=landmarker),
            patch.object(service, "process_video_frame", side_effect=lambda frame, *_args: result_by_frame(frame)),
            patch.object(service, "choose_elbow_angle", side_effect=angle_for_pose),
        ):
            response = service.assess_pushup_video(video_path, model_path, "sample.mp4")
        return response, capture, landmarker

    def test_service_composes_pose_measurements_and_phase_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "upload.mp4"
            model = Path(directory) / "pose.task"
            video.write_bytes(b"video placeholder")
            model.write_bytes(b"model placeholder")

            response, capture, landmarker = self._run_with_mocks(video, model)

        self.assertEqual(response.detected_repetitions, 1)
        self.assertEqual(response.total_frames_processed, 40)
        self.assertEqual(response.frames_with_detected_poses, 40)
        self.assertEqual(response.pose_detection_percentage, 100)
        self.assertEqual(response.detector_configuration.bottom_angle_deg, 90)
        self.assertEqual(response.detector_configuration.top_angle_deg, 160)
        self.assertEqual(len(response.accepted_repetition_timestamps), 1)
        self.assertTrue(capture.released)
        self.assertTrue(landmarker.closed)

    def test_missing_model_fails_before_opening_video(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "upload.mp4"
            video.write_bytes(b"video placeholder")
            with self.assertRaises(service.MissingPoseModelError):
                service.assess_pushup_video(video, Path(directory) / "missing.task", "upload.mp4")

    def test_processing_failure_releases_capture_and_landmarker(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "upload.mp4"
            model = Path(directory) / "pose.task"
            video.write_bytes(b"video placeholder")
            model.write_bytes(b"model placeholder")
            capture = FakeCapture(1)
            landmarker = FakeLandmarker()
            with (
                patch.object(service, "load_pose_dependencies", return_value=(object(), object(), object())),
                patch.object(
                    service,
                    "open_video",
                    return_value=(capture, service.EngineVideoMetadata(25.0, 640, 480)),
                ),
                patch.object(service, "create_video_landmarker", return_value=landmarker),
                patch.object(service, "process_video_frame", side_effect=RuntimeError("synthetic failure")),
                self.assertRaises(service.AssessmentProcessingError),
            ):
                service.assess_pushup_video(video, model, "upload.mp4")
            self.assertTrue(capture.released)
            self.assertTrue(landmarker.closed)


class PushUpAssessmentEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.model = Path(self.temp_directory.name) / "pose.task"
        self.model.write_bytes(b"model placeholder")

    def upload(self, filename: str, content: bytes) -> UploadFile:
        return UploadFile(filename=filename, file=BytesIO(content))

    def test_successful_endpoint_saves_then_removes_temporary_video(self):
        video = self.upload("sample.mp4", b"video contents")
        observed: dict[str, Path] = {}

        def fake_assess(video_path, model_path, input_filename):
            observed["path"] = video_path
            self.assertTrue(video_path.is_file())
            self.assertEqual(video_path.read_bytes(), b"video contents")
            self.assertEqual(model_path, self.model)
            self.assertEqual(input_filename, "sample.mp4")
            return response_fixture(input_filename)

        with (
            patch.object(api_v1, "get_settings", return_value=Settings(self.model, 100)),
            patch.object(api_v1, "assess_pushup_video", side_effect=fake_assess),
        ):
            response = api_v1.create_pushup_assessment(video)

        self.assertEqual(response.detected_repetitions, 0)
        self.assertFalse(observed["path"].exists())
        self.assertTrue(video.file.closed)

    def test_unsupported_extension_returns_415(self):
        video = self.upload("sample.txt", b"not a video")
        with self.assertRaises(HTTPException) as raised:
            api_v1.create_pushup_assessment(video)
        self.assertEqual(raised.exception.status_code, 415)
        self.assertTrue(video.file.closed)

    def test_oversized_upload_returns_413_and_skips_processing(self):
        video = self.upload("sample.mp4", b"1234")
        with (
            patch.object(api_v1, "get_settings", return_value=Settings(self.model, 3)),
            patch.object(api_v1, "assess_pushup_video") as assess,
            self.assertRaises(HTTPException) as raised,
        ):
            api_v1.create_pushup_assessment(video)
        self.assertEqual(raised.exception.status_code, 413)
        assess.assert_not_called()
        self.assertTrue(video.file.closed)

    def test_missing_model_returns_503(self):
        video = self.upload("sample.mp4", b"video")
        missing_model = self.model.with_name("missing.task")
        with (
            patch.object(api_v1, "get_settings", return_value=Settings(missing_model, 100)),
            self.assertRaises(HTTPException) as raised,
        ):
            api_v1.create_pushup_assessment(video)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertTrue(video.file.closed)

    def test_unreadable_video_returns_422(self):
        video = self.upload("sample.mp4", b"not actually a video")
        with (
            patch.object(api_v1, "get_settings", return_value=Settings(self.model, 100)),
            patch.object(api_v1, "assess_pushup_video", side_effect=service.InvalidVideoError("bad video")),
            self.assertRaises(HTTPException) as raised,
        ):
            api_v1.create_pushup_assessment(video)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertTrue(video.file.closed)

    def test_processing_failure_returns_500(self):
        video = self.upload("sample.mp4", b"video")
        with (
            patch.object(api_v1, "get_settings", return_value=Settings(self.model, 100)),
            patch.object(api_v1, "assess_pushup_video", side_effect=service.AssessmentProcessingError("failure")),
            patch.object(api_v1.logger, "exception"),
            self.assertRaises(HTTPException) as raised,
        ):
            api_v1.create_pushup_assessment(video)
        self.assertEqual(raised.exception.status_code, 500)
        self.assertTrue(video.file.closed)


if __name__ == "__main__":
    unittest.main()
