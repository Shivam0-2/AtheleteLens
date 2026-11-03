"""Environment-based settings for the local assessment prototype."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POSE_MODEL_PATH = (
    REPOSITORY_ROOT / "ai_engine" / "models" / "pose_landmarker_lite.task"
)
DEFAULT_MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    """Runtime settings read from the environment when handling a request."""

    pose_model_path: Path
    max_upload_size_bytes: int


def get_settings() -> Settings:
    """Read and validate AI model path and upload-size environment settings."""
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
    return Settings(model_path, max_upload_size)
