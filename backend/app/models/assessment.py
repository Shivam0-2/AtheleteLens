"""Persistent assessment metadata and response record; videos stay in file storage."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Index, JSON, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AssessmentORM(Base):
    __tablename__ = "assessments"
    __table_args__ = (
        Index("idx_assessments_athlete_submitted", "athlete_id", "submitted_at"),
    )

    assessment_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    athlete_id: Mapped[str] = mapped_column(String(200), nullable=False)
    exercise_type: Mapped[str] = mapped_column(String(64), nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    response_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    configuration_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    algorithm_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    annotated_video_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    annotated_video_url: Mapped[str | None] = mapped_column(Text, nullable=True)
