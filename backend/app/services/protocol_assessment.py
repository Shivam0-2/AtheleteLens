"""Run an AssessmentProtocol implementation through the shared pose pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from athletelens_ai.assessments.base import AssessmentProtocol
from athletelens_ai.pipeline import PoseVideoPipeline, VideoInputError

from app.schemas.assessment import ExerciseAssessmentResponse, VideoMetadata
from app.core.config import get_settings
from app.services.annotated_video import build_annotation_lines, create_annotated_video
from app.services.errors import (
    AssessmentProcessingError,
    InvalidVideoError,
    MissingPoseModelError,
)


AssessmentFactory = Callable[[Any], AssessmentProtocol]


def assess_protocol_video(
    video_path: Path,
    model_path: Path,
    input_filename: str,
    assessment_factory: AssessmentFactory,
    *,
    pipeline_factory: Callable[..., PoseVideoPipeline] = PoseVideoPipeline,
) -> ExerciseAssessmentResponse:
    """Measure and analyze a video without duplicating pose initialization.

    The factory receives shared video metadata before the first observation is
    measured. This lets image-dependent assessments, such as curl-ups, derive
    configuration from the actual uploaded frame dimensions.
    """
    video_path, model_path = Path(video_path), Path(model_path)
    if not video_path.is_file():
        raise InvalidVideoError(f"Uploaded video is not available: {video_path.name}")
    if not model_path.is_file():
        raise MissingPoseModelError(f"Pose model file is unavailable: {model_path}")

    observations = []
    measurements = []
    pose_frames: list[list[dict[str, Any]]] = []
    pose_count = 0
    assessment: AssessmentProtocol | None = None
    try:
        with pipeline_factory(video_path, model_path) as pipeline:
            metadata = pipeline.metadata
            if metadata is None:
                raise InvalidVideoError("Video metadata is unavailable")
            assessment = assessment_factory(metadata)
            for observation in pipeline:
                observations.append(observation)
                pose_frames.append(
                    observation.pose_landmark_data[0]
                    if observation.pose_landmark_data else []
                )
                pose_count += int(observation.pose_detected)
                measurements.append(assessment.measure_frame(observation))
        if not observations:
            raise InvalidVideoError("Uploaded video contains no readable frames")
        assert assessment is not None
        result = assessment.analyze(measurements, metadata)
    except (InvalidVideoError, MissingPoseModelError, AssessmentProcessingError):
        raise
    except VideoInputError as exc:
        raise InvalidVideoError("Uploaded file is not a readable video") from exc
    except Exception as exc:
        raise AssessmentProcessingError(
            f"Assessment processing failed ({type(exc).__name__})"
        ) from exc

    summary = dict(result.summary)
    events = [dict(event) for event in result.events]
    accepted = [event for event in events if event.get("event_type") in {"accepted", "completed"}]
    rejected = [event for event in events if event.get("event_type") == "rejected"]
    valid_count = sum(bool(item.quality.get("measurement_valid")) for item in measurements)
    frame_count = len(observations)
    pose_percentage = round(100 * pose_count / frame_count, 2)
    measurement_percentage = round(100 * valid_count / frame_count, 2)
    missing_landmark_counts: Counter[str] = Counter()
    for item in measurements:
        if item.quality.get("measurement_valid"):
            continue
        missing = item.quality.get("missing_landmarks")
        if isinstance(missing, str) and missing:
            missing_landmark_counts.update(part.strip() for part in missing.split(",") if part.strip())
    visibility_values = [
        float(item.quality["visibility_score"])
        for item in measurements
        if item.quality.get("visibility_score") is not None
    ]
    mean_visibility = summary.get("mean_visibility")
    if mean_visibility is None and visibility_values:
        mean_visibility = round(sum(visibility_values) / len(visibility_values), 4)

    warnings = [
        "Experimental computer-vision output; not scientifically validated.",
        "Annotated video is generated locally when processing succeeds.",
    ]
    protocol_gaps = summary.get("protocol_gaps")
    if isinstance(protocol_gaps, list):
        warnings.extend(str(item) for item in protocol_gaps)
    if result.assessment_type == "vertical_jump":
        warnings.append(str(summary.get("height_assumption", "Flight-time height assumes equal takeoff and landing height.")))
    if result.assessment_type == "standing_broad_jump" and not summary.get("physical_distance_available"):
        warnings.append("Physical distance is unavailable without valid calibration; displacement is reported in pixels.")

    excluded_summary_fields = {
        "measurements", "accepted_attempts", "rejected_attempts", "accepted_repetitions",
        "attempts_with_physical_distance",
    }
    measurement_summary = {
        key: value for key, value in summary.items() if key not in excluded_summary_fields
    }
    annotation_url = None
    try:
        _, annotation_url = create_annotated_video(
            video_path, get_settings().annotated_video_dir, pose_frames,
            build_annotation_lines(
                result.assessment_type,
                summary.get("measurements", []),
                events,
                [observation.timestamp_ms for observation in observations],
            ),
            expected_width=metadata.width, expected_height=metadata.height,
            expected_fps=metadata.fps,
        )
    except Exception as exc:
        warnings.append(f"Annotated video is unavailable ({type(exc).__name__}).")
    return ExerciseAssessmentResponse(
        exercise_type=result.assessment_type,
        processing_status="completed",
        input_filename=input_filename,
        video=VideoMetadata(fps=metadata.fps, width=metadata.width, height=metadata.height),
        total_frames_processed=frame_count,
        frames_with_detected_poses=pose_count,
        frames_with_valid_measurements=valid_count,
        pose_detection_percentage=pose_percentage,
        measurement_coverage_percentage=measurement_percentage,
        measurement_quality={
            "valid_measurement_frames": valid_count,
            "frames_missing_measurements": frame_count - valid_count,
            "measurement_coverage_percentage": measurement_percentage,
            "mean_landmark_visibility": mean_visibility,
            "missing_landmark_counts": dict(missing_landmark_counts),
            "confidence_is_probability": False,
        },
        accepted_attempt_count=len(accepted),
        accepted_attempts=accepted,
        rejected_attempt_count=len(rejected),
        rejected_attempts=rejected,
        measurement_summary=measurement_summary,
        physical_distance_attempt_count=summary.get("physical_distance_attempt_count"),
        physical_distance_available=summary.get("physical_distance_available"),
        detector_configuration=dict(result.configuration),
        annotated_video_available=annotation_url is not None,
        annotated_video_url=annotation_url,
        annotated_video_path=None,
        warnings=warnings,
    )
