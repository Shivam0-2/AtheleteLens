"""Create portable assessment history table.

Revision ID: 20261121_01
Revises:
Create Date: 2026-11-21
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20261121_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if "assessments" not in tables:
        op.create_table(
            "assessments",
            sa.Column("assessment_id", sa.Uuid(as_uuid=True), primary_key=True, nullable=False),
            sa.Column("athlete_id", sa.String(length=200), nullable=False),
            sa.Column("exercise_type", sa.String(length=64), nullable=False),
            sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("response_json", sa.JSON(), nullable=False),
            sa.Column("configuration_json", sa.JSON(), nullable=False),
            sa.Column("algorithm_version", sa.String(length=128), nullable=True),
            sa.Column("annotated_video_id", sa.String(length=32), nullable=True),
            sa.Column("annotated_video_url", sa.Text(), nullable=True),
        )
        inspector = inspect(op.get_bind())
    indexes = {item["name"] for item in inspector.get_indexes("assessments")}
    if "idx_assessments_athlete_submitted" not in indexes:
        op.create_index(
            "idx_assessments_athlete_submitted",
            "assessments",
            ["athlete_id", "submitted_at"],
        )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if "assessments" in inspector.get_table_names():
        indexes = {item["name"] for item in inspector.get_indexes("assessments")}
        if "idx_assessments_athlete_submitted" in indexes:
            op.drop_index("idx_assessments_athlete_submitted", table_name="assessments")
        op.drop_table("assessments")
