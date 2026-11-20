"""Endpoint and OpenAPI coverage for all supported exercise uploads."""

from __future__ import annotations

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from athletelens_ai.video import VideoMetadata as EngineVideoMetadata
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from app.api.v1 import router as api_v1
from app.core.config import Settings
from app.main import app
from app.schemas.assessment import ExerciseAssessmentResponse, PushUpAssessmentResponse


def exercise_response(exercise_type: str) -> ExerciseAssessmentResponse:
    return ExerciseAssessmentResponse(
        exercise_type=exercise_type,
        processing_status="completed",
        input_filename="sample.mp4",
        video={"fps": 30, "width": 640, "height": 480},
        total_frames_processed=1,
        frames_with_detected_poses=1,
        frames_with_valid_measurements=1,
        pose_detection_percentage=100,
        measurement_coverage_percentage=100,
        measurement_quality={"valid_measurement_frames": 1},
        accepted_attempt_count=0,
        accepted_attempts=[],
        rejected_attempt_count=0,
        rejected_attempts=[],
        measurement_summary={},
        detector_configuration={},
        annotated_video_available=False,
        annotated_video_path=None,
        warnings=["Experimental output."],
    )


class ExerciseAssessmentEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.model = Path(self.temp_directory.name) / "pose.task"
        self.model.write_bytes(b"model")

    @staticmethod
    def upload() -> UploadFile:
        return UploadFile(filename="sample.mp4", file=BytesIO(b"video"))

    def test_all_three_protocol_assessment_routes_build_config_and_return_response(self):
        curl_config = json.dumps({"extended_angle_deg": 150, "flexed_angle_deg": 100})
        broad_config = json.dumps({
            "reference_endpoint_a_x_px": 10,
            "reference_endpoint_a_y_px": 200,
            "reference_endpoint_b_x_px": 210,
            "reference_endpoint_b_y_px": 200,
            "reference_length_m": 2,
        })
        cases = (
            (api_v1.create_curlup_assessment, "curl_up", curl_config),
            (api_v1.create_vertical_jump_assessment, "vertical_jump", None),
            (api_v1.create_broad_jump_assessment, "standing_broad_jump", broad_config),
        )

        for endpoint, exercise_type, config_json in cases:
            with self.subTest(exercise=exercise_type):
                video = self.upload()
                observed: dict[str, object] = {}

                def fake_process(path, model, filename, assessment_factory):
                    self.assertTrue(path.is_file())
                    self.assertEqual(path.read_bytes(), b"video")
                    self.assertEqual(model, self.model)
                    self.assertEqual(filename, "sample.mp4")
                    assessment = assessment_factory(EngineVideoMetadata(30, 640, 480))
                    observed["assessment"] = assessment
                    observed["path"] = path
                    return exercise_response(exercise_type)

                with (
                    patch.object(api_v1, "get_settings", return_value=Settings(self.model, 100)),
                    patch.object(api_v1, "assess_protocol_video", side_effect=fake_process),
                ):
                    response = endpoint(video, config_json)

                self.assertEqual(response.exercise_type, exercise_type)
                self.assertEqual(response.processing_status, "completed")
                self.assertFalse(response.annotated_video_available)
                self.assertIsNone(response.annotated_video_path)
                self.assertFalse(observed["path"].exists())
                self.assertTrue(video.file.closed)

    def test_pushup_response_retains_legacy_fields_and_adds_common_metadata(self):
        video = self.upload()
        with (
            patch.object(api_v1, "get_settings", return_value=Settings(self.model, 100)),
            patch.object(api_v1, "assess_pushup_video", return_value=PushUpAssessmentResponse(
                input_filename="sample.mp4",
                video={"fps": 30, "width": 640, "height": 480},
                total_frames_processed=1,
                frames_with_detected_poses=1,
                frames_with_valid_elbow_angle=1,
                pose_detection_percentage=100,
                detected_repetitions=0,
                detector_configuration={
                    "bottom_angle_deg": 90, "top_angle_deg": 160, "hysteresis_deg": 5,
                    "confirmation_frames": 2, "min_repetition_duration_s": .25,
                    "max_repetition_duration_s": 3, "max_measurement_gap_s": .25,
                    "smoothing_window_frames": 7,
                },
                accepted_repetition_timestamps=[],
                rejected_events=[],
            )),
        ):
            response = api_v1.create_pushup_assessment(video)
        self.assertEqual(response.detected_repetitions, 0)
        self.assertEqual(response.exercise_type, "push_up")
        self.assertEqual(response.processing_status, "completed")
        self.assertTrue(video.file.closed)

    def test_invalid_or_missing_curl_config_returns_422_and_closes_upload(self):
        for config_json in (None, "{\"extended_angle_deg\": 90, \"flexed_angle_deg\": 120}"):
            video = self.upload()
            with self.subTest(config=config_json), self.assertRaises(HTTPException) as raised:
                api_v1.create_curlup_assessment(video, config_json)
            self.assertEqual(raised.exception.status_code, 422)
            self.assertTrue(video.file.closed)

    def test_partial_broad_jump_calibration_is_rejected_before_upload_processing(self):
        video = self.upload()
        with self.assertRaises(HTTPException) as raised:
            api_v1.create_broad_jump_assessment(
                video, json.dumps({"reference_endpoint_a_x_px": 20})
            )
        self.assertEqual(raised.exception.status_code, 422)
        self.assertTrue(video.file.closed)

    def test_broad_jump_calibration_contract_accepts_complete_reference_and_omission(self):
        from app.schemas.assessment_requests import BroadJumpRequestConfig

        omitted = BroadJumpRequestConfig()
        self.assertIsNone(omitted.reference_length_m)
        supplied = BroadJumpRequestConfig.model_validate_json(json.dumps({
            "reference_endpoint_a_x_px": 80,
            "reference_endpoint_a_y_px": 320,
            "reference_endpoint_b_x_px": 280,
            "reference_endpoint_b_y_px": 320,
            "reference_length_m": 2.0,
        }))
        self.assertEqual(supplied.reference_length_m, 2.0)

    def test_broad_jump_request_rejects_invalid_reference_geometry_and_length(self):
        from app.schemas.assessment_requests import BroadJumpRequestConfig

        invalid = (
            {"reference_endpoint_a_x_px": 80, "reference_endpoint_a_y_px": 320,
             "reference_endpoint_b_x_px": 80.5, "reference_endpoint_b_y_px": 320,
             "reference_length_m": 0.001},
            {"reference_endpoint_a_x_px": 80, "reference_endpoint_a_y_px": 320,
             "reference_endpoint_b_x_px": 280, "reference_endpoint_b_y_px": 320,
             "reference_length_m": 0},
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                BroadJumpRequestConfig.model_validate(payload)

    def test_openapi_documents_all_four_upload_endpoints(self):
        paths = app.openapi()["paths"]
        expected = (
            "/api/v1/assessments/pushups",
            "/api/v1/assessments/curlups",
            "/api/v1/assessments/vertical-jumps",
            "/api/v1/assessments/broad-jumps",
        )
        for path in expected:
            with self.subTest(path=path):
                operation = paths[path]["post"]
                self.assertIn("multipart/form-data", operation["requestBody"]["content"])
                self.assertTrue(operation["responses"].get("200"))
        curl_body = paths["/api/v1/assessments/curlups"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        curl_body_name = curl_body["$ref"].rsplit("/", 1)[-1]
        self.assertIn("config_json", app.openapi()["components"]["schemas"][curl_body_name]["required"])


if __name__ == "__main__":
    unittest.main()
