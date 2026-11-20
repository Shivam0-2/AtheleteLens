"""Database URL handling and SQLAlchemy engine/session helpers."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings


def database_url_for(settings: Settings) -> str:
    """Use DATABASE_URL when configured; otherwise keep local SQLite by default."""
    if settings.database_url:
        url = settings.database_url.strip()
        if url.startswith("postgres://"):
            return "postgresql+psycopg://" + url.removeprefix("postgres://")
        if url.startswith("postgresql://"):
            return "postgresql+psycopg://" + url.removeprefix("postgresql://")
        return url

    path = Path(settings.assessment_database_path).expanduser().resolve()
    return f"sqlite:///{path.as_posix()}"


def create_database_engine(database_url: str) -> Engine:
    """Create an engine for SQLite or PostgreSQL (including Supabase URLs)."""
    normalized = database_url
    if normalized.startswith("postgres://"):
        normalized = "postgresql+psycopg://" + normalized.removeprefix("postgres://")
    elif normalized.startswith("postgresql://"):
        normalized = "postgresql+psycopg://" + normalized.removeprefix("postgresql://")
    options = {"pool_pre_ping": True}
    if normalized.startswith("sqlite:"):
        options["connect_args"] = {"check_same_thread": False, "timeout": 15}
    return create_engine(normalized, **options)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
