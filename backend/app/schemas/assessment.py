"""Structured response schemas for preliminary video assessments."""

from __future__ import annotations

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
    video: VideoMetadata
    total_frames_processed: int = Field(ge=1)
    frames_with_detected_poses: int = Field(ge=0)
    frames_with_valid_elbow_angle: int = Field(ge=0)
    pose_detection_percentage: float = Field(ge=0, le=100)
    detected_repetitions: int = Field(ge=0)
    average_elbow_angle_degrees: float | None = None
    minimum_elbow_angle_degrees: float | None = None
    maximum_elbow_angle_degrees: float | None = None
    detector_configuration: DetectorConfiguration
    accepted_repetition_timestamps: list[RepetitionTimestamp]
    rejected_events: list[RejectedRepetition]
