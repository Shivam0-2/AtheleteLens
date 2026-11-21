from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import event

from app.core import config
from app.db.session import create_database_engine, database_url_for


class DatabaseDotenvConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.env_file = Path(self.temp_dir.name) / ".env"

    def test_backend_env_file_database_url_is_loaded(self) -> None:
        expected = "postgresql://pooler.example.test:5432/postgres?sslmode=require"
        self.env_file.write_text(f'DATABASE_URL="{expected}"\n', encoding="utf-8")
        with (
            patch.object(config, "BACKEND_ENV_FILE", self.env_file),
            patch.dict(os.environ, {}, clear=True),
        ):
            settings = config.get_settings()
        self.assertEqual(settings.database_url, expected)
        self.assertTrue(database_url_for(settings).startswith("postgresql+psycopg://"))

    def test_process_environment_takes_precedence_over_dotenv(self) -> None:
        self.env_file.write_text("DATABASE_URL=postgresql://file.example.test/db\n", encoding="utf-8")
        expected = "postgresql://process.example.test/db"
        with (
            patch.object(config, "BACKEND_ENV_FILE", self.env_file),
            patch.dict(os.environ, {"DATABASE_URL": expected}, clear=True),
        ):
            settings = config.get_settings()
        self.assertEqual(settings.database_url, expected)

    def test_sqlite_remains_the_fallback_when_no_url_is_configured(self) -> None:
        self.env_file.write_text("ATHLETELENS_POSE_MODEL=pose.task\n", encoding="utf-8")
        with (
            patch.object(config, "BACKEND_ENV_FILE", self.env_file),
            patch.dict(os.environ, {}, clear=True),
        ):
            settings = config.get_settings()
            resolved = database_url_for(settings)
        self.assertTrue(resolved.startswith("sqlite:///"))

    def test_postgresql_engine_has_a_finite_connection_timeout(self) -> None:
        engine = create_database_engine("postgresql+psycopg://user:pass@example.test/db")
        captured: dict[str, object] = {}

        def capture_connect_args(_dialect, _record, _args, kwargs) -> None:
            captured.update(kwargs)
            raise RuntimeError("stop before opening a network connection")

        event.listen(engine, "do_connect", capture_connect_args)
        try:
            with self.assertRaisesRegex(RuntimeError, "stop before opening"):
                engine.connect()
        finally:
            engine.dispose()
        self.assertEqual(captured.get("connect_timeout"), 8)


if __name__ == "__main__":
    unittest.main()
