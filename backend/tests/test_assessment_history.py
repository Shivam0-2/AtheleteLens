"""Persistence and ownership-scoping tests for assessment history."""

from __future__ import annotations

import tempfile
import unittest
import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from io import BytesIO
from starlette.datastructures import UploadFile

from app.api.v1 import router as api_v1
from app.core.config import Settings
from app.db.migrations import upgrade_database
from app.db.session import database_url_for
from app.db.copy_history import copy_assessments
from app.schemas.assessment import PushUpAssessmentResponse
from app.services.assessment_store import get_assessment, list_assessments, save_assessment


def pushup_response() -> PushUpAssessmentResponse:
    return PushUpAssessmentResponse(
        input_filename="real-sample.mp4",
        video={"fps": 30, "width": 640, "height": 480},
        total_frames_processed=2,
        frames_with_detected_poses=2,
        frames_with_valid_elbow_angle=2,
        pose_detection_percentage=100,
        detected_repetitions=1,
        detector_configuration={
            "bottom_angle_deg": 90, "top_angle_deg": 160, "hysteresis_deg": 5,
            "confirmation_frames": 2, "min_repetition_duration_s": .25,
            "max_repetition_duration_s": 3, "max_measurement_gap_s": .25,
            "smoothing_window_frames": 7,
        },
        accepted_repetition_timestamps=[],
        rejected_events=[],
        annotated_video_available=True,
        annotated_video_url="/api/v1/assessments/annotated-videos/" + "a" * 32,
    )


class AssessmentHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.database = Path(self.temp_dir.name) / "history.sqlite3"
        self.database_url = database_url_for(Settings(
            Path(self.temp_dir.name) / "pose.task", 1024,
            assessment_database_path=self.database,
        ))
        upgrade_database(self.database_url)

    def test_persistence_round_trips_actual_response_configuration_and_video_id(self):
        response = pushup_response()
        record = save_assessment(self.database_url, "athlete-one", response, {"requested": "default"})
        loaded = get_assessment(self.database_url, record.assessment_id)
        self.assertEqual(loaded, record)
        self.assertEqual(record.response["detected_repetitions"], 1)
        self.assertEqual(record.configuration["detector"]["top_angle_deg"], 160)
        self.assertEqual(record.annotated_video_id, "a" * 32)
        self.assertIsNone(record.algorithm_version)

    def test_history_is_associated_with_the_requested_demo_user(self):
        save_assessment(self.database_url, "athlete-one", pushup_response(), {})
        save_assessment(self.database_url, "athlete-two", pushup_response(), {})
        self.assertEqual([row.athlete_id for row in list_assessments(self.database_url, "athlete-one")], ["athlete-one"])
        self.assertEqual(len(list_assessments(self.database_url)), 2)

    def test_missing_record_returns_none(self):
        self.assertIsNone(get_assessment(self.database_url, "f" * 32))

    def test_upload_endpoint_persists_success_and_exposes_record_id_without_changing_body(self):
        model = Path(self.temp_dir.name) / "pose.task"
        model.write_bytes(b"model")
        settings = Settings(model, 1024, assessment_database_path=self.database)
        with (
            patch.object(api_v1, "get_settings", return_value=settings),
            patch.object(api_v1, "assess_pushup_video", return_value=pushup_response()),
        ):
            response = api_v1.create_pushup_assessment(
                UploadFile(filename="sample.mp4", file=BytesIO(b"video bytes")),
                x_demo_user_id="athlete-http",
            )
            history = api_v1.assessment_history("athlete-http")
            record_id = history[0]["assessment_id"]
            detail = api_v1.assessment_detail(record_id, "athlete-http")
        self.assertEqual(response.exercise_type, "push_up")
        self.assertNotIn("assessment_id", response.model_dump())
        self.assertEqual(detail["response"]["exercise_type"], "push_up")
        self.assertEqual(detail["athlete_id"], "athlete-http")

    def test_detail_is_not_returned_for_another_demo_user(self):
        record = save_assessment(self.database_url, "athlete-owner", pushup_response(), {})
        settings = Settings(Path(self.temp_dir.name) / "unused.task", 1024,
                            assessment_database_path=self.database)
        with patch.object(api_v1, "get_settings", return_value=settings):
            with self.assertRaises(HTTPException) as raised:
                api_v1.assessment_detail(record.assessment_id, "other-athlete")
        self.assertEqual(raised.exception.status_code, 404)

    def test_alembic_baseline_keeps_rows_in_the_preexisting_sqlite_schema(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.sqlite3"
        identifier = uuid.uuid4().hex
        with closing(sqlite3.connect(legacy_path)) as connection:
            with connection:
                connection.execute("""CREATE TABLE assessments (
                    assessment_id TEXT PRIMARY KEY, athlete_id TEXT NOT NULL,
                    exercise_type TEXT NOT NULL, submitted_at TEXT NOT NULL,
                    response_json TEXT NOT NULL, configuration_json TEXT NOT NULL,
                    algorithm_version TEXT, annotated_video_id TEXT, annotated_video_url TEXT
                )""")
                connection.execute(
                    "INSERT INTO assessments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (identifier, "legacy-athlete", "push_up", "2026-11-21T10:00:00+00:00",
                     json.dumps(pushup_response().model_dump(mode="json")), json.dumps({"legacy": True}),
                     None, None, None),
                )
        url = database_url_for(Settings(Path("pose.task"), 1024, assessment_database_path=legacy_path))
        upgrade_database(url)
        migrated = get_assessment(url, identifier)
        self.assertIsNotNone(migrated)
        self.assertEqual(migrated.athlete_id, "legacy-athlete")
        self.assertEqual(migrated.response["detected_repetitions"], 1)

    def test_database_url_supports_sqlite_and_supabase_postgresql_schemes(self):
        from app.db.session import create_database_engine

        sqlite_engine = create_database_engine(self.database_url)
        try:
            with sqlite_engine.connect() as connection:
                self.assertEqual(connection.exec_driver_sql("SELECT 1").scalar_one(), 1)
        finally:
            sqlite_engine.dispose()
        postgres_engine = create_database_engine(
            "postgresql://postgres.project:secret@pooler.example.test:5432/postgres?sslmode=require"
        )
        try:
            self.assertEqual(postgres_engine.dialect.name, "postgresql")
            self.assertEqual(postgres_engine.dialect.driver, "psycopg")
        finally:
            postgres_engine.dispose()

    def test_copy_history_preserves_record_fields_and_is_idempotent(self):
        target_path = Path(self.temp_dir.name) / "target.sqlite3"
        source_url = self.database_url
        target_url = database_url_for(Settings(Path("pose.task"), 1024, assessment_database_path=target_path))
        original = save_assessment(source_url, "athlete-copy", pushup_response(), {"source": True})
        self.assertEqual(copy_assessments(source_url, target_url), (1, 0))
        self.assertEqual(copy_assessments(source_url, target_url), (0, 1))
        copied = get_assessment(target_url, original.assessment_id)
        self.assertEqual(copied, original)


if __name__ == "__main__":
    unittest.main()
