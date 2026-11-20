"""One-time, idempotent transfer of assessment rows between configured databases."""

from __future__ import annotations

import argparse

from sqlalchemy import select

from app.core.config import get_settings
from app.db.migrations import upgrade_database
from app.db.session import create_database_engine, create_session_factory, database_url_for
from app.models.assessment import AssessmentORM


def copy_assessments(source_url: str, target_url: str) -> tuple[int, int]:
    """Copy response records, preserving IDs and JSON; never copies video files."""
    if source_url == target_url:
        raise ValueError("Source and target DATABASE_URL values must be different.")
    upgrade_database(source_url)
    upgrade_database(target_url)
    source_engine = create_database_engine(source_url)
    target_engine = create_database_engine(target_url)
    copied = skipped = 0
    try:
        source_factory = create_session_factory(source_engine)
        target_factory = create_session_factory(target_engine)
        with source_factory() as source, target_factory() as target:
            rows = source.scalars(select(AssessmentORM)).all()
            for row in rows:
                if target.get(AssessmentORM, row.assessment_id) is not None:
                    skipped += 1
                    continue
                target.add(AssessmentORM(
                    assessment_id=row.assessment_id,
                    athlete_id=row.athlete_id,
                    exercise_type=row.exercise_type,
                    submitted_at=row.submitted_at,
                    response_json=row.response_json,
                    configuration_json=row.configuration_json,
                    algorithm_version=row.algorithm_version,
                    annotated_video_id=row.annotated_video_id,
                    annotated_video_url=row.annotated_video_url,
                ))
                copied += 1
            target.commit()
    finally:
        source_engine.dispose()
        target_engine.dispose()
    return copied, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", required=True, help="Existing SQLite or PostgreSQL DATABASE_URL")
    arguments = parser.parse_args()
    target_url = database_url_for(get_settings())
    copied, skipped = copy_assessments(arguments.source_url, target_url)
    print(f"Copied {copied} assessment record(s); skipped {skipped} already-present record(s).")
    print("Annotated MP4 files are not copied; move them separately to the configured file-storage directory.")


if __name__ == "__main__":
    main()
