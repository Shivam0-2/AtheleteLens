"""Programmatic Alembic upgrade helper used by the container entrypoint and tests."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config


def upgrade_database(database_url: str) -> None:
    backend_root = Path(__file__).resolve().parents[2]
    config = Config(str(backend_root / "alembic.ini"))
    config.attributes["database_url"] = database_url
    command.upgrade(config, "head")
