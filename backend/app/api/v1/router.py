"""Version 1 API routes."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.schemas.assessment import PushUpAssessmentResponse
from app.services.pushup_assessment import (
    AssessmentProcessingError,
    InvalidVideoError,
    MissingPoseModelError,
    assess_pushup_video,
)

router = APIRouter()
logger = logging.getLogger(__name__)
SUPPORTED_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
UPLOAD_CHUNK_SIZE = 1024 * 1024


@router.get("/health", tags=["health"])
async def versioned_health_check() -> dict[str, str]:
    """Report that the version 1 API router is responding."""
    return {"status": "healthy", "api_version": "v1"}


@router.post(
    "/assessments/pushups",
    response_model=PushUpAssessmentResponse,
    status_code=status.HTTP_200_OK,
    tags=["assessments"],
    summary="Run a preliminary push-up assessment on an uploaded video",
)
def create_pushup_assessment(
    video: UploadFile = File(..., description="Video file to assess"),
) -> PushUpAssessmentResponse:
    """Process one uploaded video synchronously in FastAPI's worker pool."""
    filename = (video.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        video.file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported video format. Use MP4, MOV, AVI, MKV, M4V, or WebM.",
        )

    try:
        settings = get_settings()
    except ValueError as exc:
        video.file.close()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    if not settings.pose_model_path.is_file():
        video.file.close()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pose Landmarker model is unavailable on the server.",
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, prefix="athletelens-upload-", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            total_bytes = 0
            while chunk := video.file.read(UPLOAD_CHUNK_SIZE):
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_size_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=(
                            "Video exceeds the configured upload limit of "
                            f"{settings.max_upload_size_bytes} bytes."
                        ),
                    )
                temporary_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded video is empty.",
            )

        return assess_pushup_video(
            video_path=temporary_path,
            model_path=settings.pose_model_path,
            input_filename=filename,
        )
    except HTTPException:
        raise
    except MissingPoseModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pose Landmarker model is unavailable on the server.",
        ) from exc
    except InvalidVideoError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except AssessmentProcessingError as exc:
        logger.exception("Push-up video assessment failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Video assessment could not be completed.",
        ) from exc
    except OSError as exc:
        logger.exception("Push-up upload could not be stored or read")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Uploaded video could not be processed.",
        ) from exc
    finally:
        video.file.close()
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temporary upload %s", temporary_path)
