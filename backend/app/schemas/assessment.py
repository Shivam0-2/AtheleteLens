"""Structured response schemas for preliminary video assessments."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VideoMetadata(BaseModel):
    """Basic metadata for the submitted video."""

    model_config = ConfigDict(extra="forbid")

    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class DetectorConfiguration(BaseModel):
    """Thresholds and timing parameters used by the phase detector."""

    model_config = ConfigDict(extra="forbid")

    bottom_angle_deg: float
    top_angle_deg: float
    hysteresis_deg: float
    confirmation_frames: int
    min_repetition_duration_s: float
    max_repetition_duration_s: float
    max_measurement_gap_s: float
    smoothing_window_frames: int


class RepetitionTimestamp(BaseModel):
    """A phase-detector completion event."""

    repetition_number: int = Field(ge=1)
    start_timestamp_ms: int = Field(ge=0)
    end_timestamp_ms: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)


class RejectedRepetition(BaseModel):
    """An incomplete or invalid phase-detector cycle."""

    reason: str
    start_timestamp_ms: int | None = Field(default=None, ge=0)
    end_timestamp_ms: int = Field(ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)


class PushUpAssessmentResponse(BaseModel):
    """Preliminary, unvalidated measurements from one uploaded video."""

    status: str = "preliminary_unvalidated"
    assessment_notice: str = (
        "Experimental computer-vision output; not scientifically validated."
    )
    input_filename: str
    exercise_type: Literal["push_up"] = "push_up"
    processing_status: Literal["completed"] = "completed"
    video: VideoMetadata
    total_frames_processed: int = Field(ge=1)
    frames_with_detected_poses: int = Field(ge=0)
    frames_with_valid_elbow_angle: int = Field(ge=0)
    measurement_coverage_percentage: float = Field(default=0, ge=0, le=100)
    measurement_quality: dict[str, Any] = Field(default_factory=dict)
    pose_detection_percentage: float = Field(ge=0, le=100)
    detected_repetitions: int = Field(ge=0)
    average_elbow_angle_degrees: float | None = None
    minimum_elbow_angle_degrees: float | None = None
    maximum_elbow_angle_degrees: float | None = None
    detector_configuration: DetectorConfiguration
    accepted_repetition_timestamps: list[RepetitionTimestamp]
    rejected_events: list[RejectedRepetition]
    accepted_attempt_count: int = Field(default=0, ge=0)
    accepted_attempts: list[dict[str, Any]] = Field(default_factory=list)
    rejected_attempt_count: int = Field(default=0, ge=0)
    rejected_attempts: list[dict[str, Any]] = Field(default_factory=list)
    measurement_summary: dict[str, Any] = Field(default_factory=dict)
    annotated_video_available: bool = False
    annotated_video_url: str | None = None
    annotated_video_path: str | None = None
    warnings: list[str] = Field(default_factory=lambda: [
        "Experimental computer-vision output; not scientifically validated.",
        "Annotated video is generated locally when processing succeeds.",
    ])


class ExerciseAssessmentResponse(BaseModel):
    """Shared structured response for protocol-based exercise assessments."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["preliminary_unvalidated"] = "preliminary_unvalidated"
    processing_status: Literal["completed"]
    assessment_notice: str = "Experimental computer-vision output; not scientifically validated."
    input_filename: str
    exercise_type: Literal["curl_up", "vertical_jump", "standing_broad_jump"]
    video: VideoMetadata
    total_frames_processed: int = Field(ge=1)
    frames_with_detected_poses: int = Field(ge=0)
    frames_with_valid_measurements: int = Field(ge=0)
    pose_detection_percentage: float = Field(ge=0, le=100)
    measurement_coverage_percentage: float = Field(ge=0, le=100)
    measurement_quality: dict[str, Any]
    accepted_attempt_count: int = Field(ge=0)
    accepted_attempts: list[dict[str, Any]]
    rejected_attempt_count: int = Field(ge=0)
    rejected_attempts: list[dict[str, Any]]
    measurement_summary: dict[str, Any]
    physical_distance_attempt_count: int | None = Field(default=None, ge=0)
    physical_distance_available: bool | None = None
    detector_configuration: dict[str, Any]
    annotated_video_available: bool
    annotated_video_url: str | None = None
    annotated_video_path: str | None
    warnings: list[str]
