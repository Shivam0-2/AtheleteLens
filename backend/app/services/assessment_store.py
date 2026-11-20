"""SQLAlchemy repository for persisted assessment history."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select

from app.models.assessment import AssessmentORM
from app.db.session import create_database_engine, create_session_factory


class AssessmentRecord(BaseModel):
    assessment_id: str
    athlete_id: str
    exercise_type: str
    submitted_at: str
    response: dict[str, Any]
    configuration: dict[str, Any]
    algorithm_version: str | None
    annotated_video_id: str | None
    annotated_video_url: str | None


def _to_record(row: AssessmentORM) -> AssessmentRecord:
    timestamp = row.submitted_at
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return AssessmentRecord(
        assessment_id=row.assessment_id.hex,
        athlete_id=row.athlete_id,
        exercise_type=row.exercise_type,
        submitted_at=timestamp.isoformat(),
        response=row.response_json,
        configuration=row.configuration_json,
        algorithm_version=row.algorithm_version,
        annotated_video_id=row.annotated_video_id,
        annotated_video_url=row.annotated_video_url,
    )


def save_assessment(
    database_url: str,
    athlete_id: str,
    response: BaseModel,
    configuration: dict[str, Any],
) -> AssessmentRecord:
    """Persist a complete response JSON document and its configuration."""
    payload = response.model_dump(mode="json")
    annotated_url = payload.get("annotated_video_url")
    annotated_id = None
    if isinstance(annotated_url, str):
        tail = annotated_url.rstrip("/").rsplit("/", 1)[-1]
        if len(tail) == 32 and all(char in "0123456789abcdef" for char in tail):
            annotated_id = tail
    actual_configuration = {
        "request": configuration,
        "detector": payload.get("detector_configuration", {}),
    }
    row = AssessmentORM(
        assessment_id=uuid.uuid4(),
        athlete_id=athlete_id,
        exercise_type=str(payload["exercise_type"]),
        submitted_at=datetime.now(UTC),
        response_json=payload,
        configuration_json=actual_configuration,
        algorithm_version=None,
        annotated_video_id=annotated_id,
        annotated_video_url=annotated_url if isinstance(annotated_url, str) else None,
    )
    engine = create_database_engine(database_url)
    try:
        with create_session_factory(engine)() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return _to_record(row)
    finally:
        engine.dispose()


def list_assessments(database_url: str, athlete_id: str | None = None) -> list[AssessmentRecord]:
    engine = create_database_engine(database_url)
    try:
        with create_session_factory(engine)() as session:
            statement = select(AssessmentORM).order_by(
                AssessmentORM.submitted_at.desc(), AssessmentORM.assessment_id.desc()
            )
            if athlete_id is not None:
                statement = statement.where(AssessmentORM.athlete_id == athlete_id)
            return [_to_record(row) for row in session.scalars(statement).all()]
    finally:
        engine.dispose()


def get_assessment(database_url: str, assessment_id: str) -> AssessmentRecord | None:
    try:
        identifier = uuid.UUID(hex=assessment_id)
    except (ValueError, AttributeError):
        return None
    engine = create_database_engine(database_url)
    try:
        with create_session_factory(engine)() as session:
            row = session.get(AssessmentORM, identifier)
            return _to_record(row) if row is not None else None
    finally:
        engine.dispose()
