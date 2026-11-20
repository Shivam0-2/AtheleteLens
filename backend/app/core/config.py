"""Environment-based settings for the local assessment prototype."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POSE_MODEL_PATH = (
    REPOSITORY_ROOT / "ai_engine" / "models" / "pose_landmarker_lite.task"
)
DEFAULT_MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024
DEFAULT_ANNOTATED_VIDEO_DIR = REPOSITORY_ROOT / "backend" / "storage" / "annotated_videos"
DEFAULT_ASSESSMENT_DATABASE_PATH = REPOSITORY_ROOT / "backend" / "storage" / "assessments.sqlite3"
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


@dataclass(frozen=True)
class Settings:
    """Runtime settings read from the environment when handling a request."""

    pose_model_path: Path
    max_upload_size_bytes: int
    annotated_video_dir: Path = DEFAULT_ANNOTATED_VIDEO_DIR
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    assessment_database_path: Path = DEFAULT_ASSESSMENT_DATABASE_PATH
    database_url: str | None = None


def get_settings() -> Settings:
    """Read and validate AI model path and upload-size environment settings."""
    # Load the backend-local dotenv file independently of the shell's working
    # directory. Existing process environment values intentionally take priority.
    load_dotenv(dotenv_path=BACKEND_ENV_FILE, override=False)
    model_path = Path(
        os.getenv("ATHLETELENS_POSE_MODEL", str(DEFAULT_POSE_MODEL_PATH))
    ).expanduser()
    raw_max_size = os.getenv(
        "ATHLETELENS_MAX_UPLOAD_SIZE_BYTES", str(DEFAULT_MAX_UPLOAD_SIZE_BYTES)
    )
    try:
        max_upload_size = int(raw_max_size)
    except ValueError as exc:
        raise ValueError(
            "ATHLETELENS_MAX_UPLOAD_SIZE_BYTES must be a positive integer"
        ) from exc
    if max_upload_size < 1:
        raise ValueError("ATHLETELENS_MAX_UPLOAD_SIZE_BYTES must be a positive integer")
    output_dir = Path(os.getenv(
        "ATHLETELENS_ANNOTATED_VIDEO_DIR", str(DEFAULT_ANNOTATED_VIDEO_DIR)
    )).expanduser()
    raw_origins = os.getenv("ATHLETELENS_CORS_ORIGINS")
    origins = tuple(dict.fromkeys(
        origin.strip().rstrip("/")
        for origin in raw_origins.split(",")
        if origin.strip()
    )) if raw_origins is not None else DEFAULT_CORS_ORIGINS
    for origin in origins:
        parsed = urlsplit(origin)
        invalid_port = False
        try:
            parsed.port
        except ValueError:
            invalid_port = True
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc
                or not parsed.hostname or "*" in parsed.netloc or invalid_port
                or parsed.path or parsed.query or parsed.fragment or parsed.username
                or parsed.password):
            raise ValueError(
                "ATHLETELENS_CORS_ORIGINS must contain exact http(s) origins; "
                "wildcards, paths, queries and credentials are not allowed"
            )
    database_path = Path(os.getenv(
        "ATHLETELENS_ASSESSMENT_DATABASE", str(DEFAULT_ASSESSMENT_DATABASE_PATH)
    )).expanduser()
    database_url = os.getenv("DATABASE_URL")
    return Settings(model_path, max_upload_size, output_dir, origins, database_path, database_url)
