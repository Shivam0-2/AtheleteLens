"""Version 1 routes for preliminary AthleteLens assessments."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from athletelens_ai.assessments.broad_jump import BroadJumpAssessment, BroadJumpConfig
from athletelens_ai.assessments.calibration import GroundReferenceCalibration
from athletelens_ai.assessments.curl_up import CurlUpAssessment, CurlUpConfig
from athletelens_ai.assessments.vertical_jump import VerticalJumpAssessment, VerticalJumpConfig
from athletelens_ai.video import VideoMetadata as EngineVideoMetadata
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.schemas.assessment import ExerciseAssessmentResponse, PushUpAssessmentResponse
from app.schemas.assessment_requests import (
    BroadJumpRequestConfig,
    CurlUpRequestConfig,
    PushUpRequestConfig,
    VerticalJumpRequestConfig,
)
from app.services.errors import (
    AssessmentProcessingError,
    InvalidVideoError,
    MissingPoseModelError,
)
from app.services.protocol_assessment import assess_protocol_video
from app.services.pushup_assessment import assess_pushup_video

router = APIRouter()
logger = logging.getLogger(__name__)
SUPPORTED_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
UPLOAD_CHUNK_SIZE = 1024 * 1024
UploadProcessor = Callable[[Path, Path, str], Any]


def _request_config(
    config_json: str | None,
    schema: type[BaseModel],
    *,
    required: bool = False,
    video: UploadFile | None = None,
) -> BaseModel:
    try:
        if config_json is None:
            if required:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Assessment configuration is required for this exercise.",
                )
            return schema()
        return schema.model_validate_json(config_json)
    except HTTPException:
        if video is not None:
            video.file.close()
        raise
    except ValidationError as exc:
        if video is not None:
            video.file.close()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=exc.errors(include_url=False),
        ) from exc


def _handle_video_upload(video: UploadFile, processor: UploadProcessor) -> Any:
    """Validate, stage, process, and always remove one multipart video upload."""
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

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
                        detail=("Video exceeds the configured upload limit of "
                                f"{settings.max_upload_size_bytes} bytes."),
                    )
                temporary_file.write(chunk)
        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Uploaded video is empty.",
            )
        return processor(temporary_path, settings.pose_model_path, filename)
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
        logger.exception("Assessment processing failed for uploaded video")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Video assessment could not be completed.",
        ) from exc
    except OSError as exc:
        logger.exception("Assessment upload could not be stored or read")
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
    config_json: Annotated[str | None, Form(description="Optional JSON push-up detector configuration.")] = None,
) -> PushUpAssessmentResponse:
    options = _request_config(config_json, PushUpRequestConfig, video=video)
    assert isinstance(options, PushUpRequestConfig)
    if config_json is None:
        processor = lambda path, model, filename: assess_pushup_video(path, model, filename)
    else:
        from athletelens_ai.assessments.push_up import DetectorConfig

        detector_config = DetectorConfig(**options.model_dump())
        processor = lambda path, model, filename: assess_pushup_video(
            path, model, filename, detector_config=detector_config
        )
    return _handle_video_upload(video, processor)


@router.post(
    "/assessments/curlups",
    response_model=ExerciseAssessmentResponse,
    status_code=status.HTTP_200_OK,
    tags=["assessments"],
    summary="Run an experimental curl-up assessment",
)
def create_curlup_assessment(
    video: Annotated[UploadFile, File(description="Video file to assess")],
    config_json: Annotated[str, Form(
        description="JSON options; extended_angle_deg and flexed_angle_deg are required because the repository has no standardized curl-up thresholds.",
    )],
) -> ExerciseAssessmentResponse:
    options = _request_config(config_json, CurlUpRequestConfig, required=True, video=video)
    assert isinstance(options, CurlUpRequestConfig)

    def factory(metadata: EngineVideoMetadata) -> CurlUpAssessment:
        config = CurlUpConfig(
            image_aspect_ratio=metadata.width / metadata.height,
            **options.model_dump(),
        )
        return CurlUpAssessment(config)

    return _handle_video_upload(
        video,
        lambda path, model, filename: assess_protocol_video(path, model, filename, factory),
    )


@router.post(
    "/assessments/vertical-jumps",
    response_model=ExerciseAssessmentResponse,
    status_code=status.HTTP_200_OK,
    tags=["assessments"],
    summary="Run an experimental standing vertical-jump assessment",
)
def create_vertical_jump_assessment(
    video: UploadFile = File(..., description="Video file to assess"),
    config_json: Annotated[str | None, Form(description="Optional JSON vertical-jump detector configuration.")] = None,
) -> ExerciseAssessmentResponse:
    options = _request_config(config_json, VerticalJumpRequestConfig, video=video)
    assert isinstance(options, VerticalJumpRequestConfig)
    config = VerticalJumpConfig(**options.model_dump())
    factory = lambda _metadata: VerticalJumpAssessment(config)
    return _handle_video_upload(
        video,
        lambda path, model, filename: assess_protocol_video(path, model, filename, factory),
    )


@router.post(
    "/assessments/broad-jumps",
    response_model=ExerciseAssessmentResponse,
    status_code=status.HTTP_200_OK,
    tags=["assessments"],
    summary="Run an experimental standing broad-jump assessment",
)
def create_broad_jump_assessment(
    video: UploadFile = File(..., description="Video file to assess"),
    config_json: Annotated[str | None, Form(
        description="Optional JSON broad-jump configuration, including a ground-reference endpoint pair and its known length in meters.",
    )] = None,
) -> ExerciseAssessmentResponse:
    options = _request_config(config_json, BroadJumpRequestConfig, video=video)
    assert isinstance(options, BroadJumpRequestConfig)
    values = options.model_dump()
    reference = None
    if options.reference_length_m is not None:
        reference = GroundReferenceCalibration(
            endpoint_a_px=(options.reference_endpoint_a_x_px, options.reference_endpoint_a_y_px),
            endpoint_b_px=(options.reference_endpoint_b_x_px, options.reference_endpoint_b_y_px),
            known_length_m=options.reference_length_m,
        )
    detector_values = {
        key: value for key, value in values.items()
        if not key.startswith("reference_endpoint_") and key != "reference_length_m"
    }
    config = BroadJumpConfig(**detector_values, ground_reference=reference)
    factory = lambda _metadata: BroadJumpAssessment(config)
    return _handle_video_upload(
        video,
        lambda path, model, filename: assess_protocol_video(path, model, filename, factory),
    )
