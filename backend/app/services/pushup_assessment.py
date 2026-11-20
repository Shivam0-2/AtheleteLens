"""Compose shared video and pose helpers with the phase-based detector."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from athletelens_ai.assessments.push_up import (
    DetectorConfig,
    PhaseBasedPushUpDetector,
    choose_elbow_angle,
)
from athletelens_ai.pose import (
    create_video_landmarker,
    load_pose_dependencies,
    process_video_frame,
)
from athletelens_ai.pipeline import PoseVideoPipeline, VideoInputError
from athletelens_ai.signals import rolling_median
from athletelens_ai.video import VideoMetadata as EngineVideoMetadata
from athletelens_ai.video import open_video

from app.schemas.assessment import (
    DetectorConfiguration,
    PushUpAssessmentResponse,
    RejectedRepetition,
    RepetitionTimestamp,
    VideoMetadata,
)
from app.core.config import get_settings
from app.services.annotated_video import build_annotation_lines, create_annotated_video
from app.services.errors import (
    AssessmentProcessingError,
    AssessmentServiceError,
    InvalidVideoError,
    MissingPoseModelError,
)


MIN_LANDMARK_VISIBILITY = 0.5
SMOOTHING_WINDOW_FRAMES = 7


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def assess_pushup_video(
    video_path: Path,
    model_path: Path,
    input_filename: str,
    detector_config: DetectorConfig | None = None,
) -> PushUpAssessmentResponse:
    """Analyze a temporary video path using the existing phase detector.

    Each frame is run through MediaPipe VIDEO mode. Raw elbow angles are
    collected, smoothed with the shared missing-aware seven-frame median, and
    passed to exactly one ``PhaseBasedPushUpDetector`` instance.
    """
    video_path = Path(video_path)
    model_path = Path(model_path)
    if not video_path.is_file():
        raise InvalidVideoError(f"Uploaded video is not available: {video_path.name}")
    if not model_path.is_file():
        raise MissingPoseModelError(f"Pose model file is unavailable: {model_path}")

    config = detector_config or DetectorConfig()
    try:
        config.validate()
    except ValueError as exc:
        raise AssessmentProcessingError(f"Invalid detector configuration: {exc}") from exc

    try:
        try:
            pipeline = PoseVideoPipeline(
                video_path,
                model_path,
                dependency_loader=load_pose_dependencies,
                video_opener=open_video,
                landmarker_factory=create_video_landmarker,
                frame_processor=process_video_frame,
            )
            timestamps: list[int] = []
            raw_angles: list[float | None] = []
            pose_frames: list[list[dict[str, Any]]] = []
            frames_with_poses = 0
            with pipeline:
                metadata = pipeline.metadata
                for observation in pipeline:
                    pose_frames.append(
                        observation.pose_landmark_data[0]
                        if observation.pose_landmark_data else []
                    )
                    if observation.pose_detected:
                        frames_with_poses += 1
                        angle, _side, _left, _right, _fields = choose_elbow_angle(
                            observation.pose_landmarks[0], MIN_LANDMARK_VISIBILITY
                        )
                        if angle is not None and not math.isfinite(angle):
                            angle = None
                    else:
                        angle = None
                    timestamps.append(observation.timestamp_ms)
                    raw_angles.append(angle)
        except VideoInputError as exc:
            raise InvalidVideoError("Uploaded file is not a readable video") from exc
        except AssessmentServiceError:
            raise
        except Exception as exc:
            raise AssessmentProcessingError(
                f"Pose processing failed ({type(exc).__name__})"
            ) from exc

        if not timestamps:
            raise InvalidVideoError("Uploaded video contains no readable frames")

        smoothed_angles = rolling_median(raw_angles, SMOOTHING_WINDOW_FRAMES)
        detector = PhaseBasedPushUpDetector(config)
        detector_rows: list[dict[str, Any]] = []
        for index, (timestamp_ms, angle) in enumerate(
            zip(timestamps, smoothed_angles), start=1
        ):
            frame_row = detector.update(index, timestamp_ms, angle)
            frame_row["frame_number"] = index - 1
            detector_rows.append(frame_row)
        frame_interval_ms = (
            timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else 0
        )
        detector.finish(timestamps[-1] + frame_interval_ms)

        accepted = [
            RepetitionTimestamp(
                repetition_number=int(event["repetition_number"]),
                start_timestamp_ms=int(event["start_timestamp_ms"]),
                end_timestamp_ms=int(event["end_timestamp_ms"]),
                duration_seconds=float(event["duration_seconds"]),
            )
            for event in detector.events
            if event["event_type"] == "completed"
        ]
        rejected = [
            RejectedRepetition(
                reason=str(event["reason"]),
                start_timestamp_ms=_optional_int(event.get("start_timestamp_ms")),
                end_timestamp_ms=int(event["end_timestamp_ms"]),
                duration_seconds=_optional_float(event.get("duration_seconds")),
            )
            for event in detector.events
            if event["event_type"] == "rejected"
        ]
        valid_angles = [angle for angle in raw_angles if angle is not None]
        frames_processed = len(timestamps)
        accepted_payload = [item.model_dump() for item in accepted]
        rejected_payload = [item.model_dump() for item in rejected]
        measurement_coverage = round(len(valid_angles) * 100.0 / frames_processed, 2)
        annotation_url = None
        annotation_warnings: list[str] = []
        try:
            _, annotation_url = create_annotated_video(
                video_path, get_settings().annotated_video_dir, pose_frames,
                build_annotation_lines("push_up", detector_rows, list(detector.events), timestamps),
                expected_width=metadata.width, expected_height=metadata.height,
                expected_fps=metadata.fps,
            )
        except Exception as exc:
            annotation_warnings.append(f"Annotated video is unavailable ({type(exc).__name__}).")
        return PushUpAssessmentResponse(
            input_filename=input_filename,
            video=VideoMetadata(
                fps=metadata.fps, width=metadata.width, height=metadata.height
            ),
            total_frames_processed=frames_processed,
            frames_with_detected_poses=frames_with_poses,
            frames_with_valid_elbow_angle=len(valid_angles),
            exercise_type="push_up",
            processing_status="completed",
            measurement_coverage_percentage=measurement_coverage,
            measurement_quality={
                "valid_measurement_frames": len(valid_angles),
                "frames_missing_measurements": frames_processed - len(valid_angles),
                "measurement_coverage_percentage": measurement_coverage,
                "mean_landmark_visibility": None,
                "missing_landmark_counts": {},
                "confidence_is_probability": False,
            },
            accepted_attempt_count=len(accepted),
            accepted_attempts=accepted_payload,
            rejected_attempt_count=len(rejected),
            rejected_attempts=rejected_payload,
            measurement_summary={
                "average_elbow_angle_degrees": round(sum(valid_angles) / len(valid_angles), 3) if valid_angles else None,
                "minimum_elbow_angle_degrees": round(min(valid_angles), 3) if valid_angles else None,
                "maximum_elbow_angle_degrees": round(max(valid_angles), 3) if valid_angles else None,
            },
            annotated_video_available=annotation_url is not None,
            annotated_video_url=annotation_url,
            annotated_video_path=None,
            warnings=[
                "Experimental computer-vision output; not scientifically validated.",
                *annotation_warnings,
            ],
            pose_detection_percentage=round(
                frames_with_poses * 100.0 / frames_processed, 2
            ),
            detected_repetitions=detector.repetitions,
            average_elbow_angle_degrees=(
                round(sum(valid_angles) / len(valid_angles), 3)
                if valid_angles
                else None
            ),
            minimum_elbow_angle_degrees=(
                round(min(valid_angles), 3) if valid_angles else None
            ),
            maximum_elbow_angle_degrees=(
                round(max(valid_angles), 3) if valid_angles else None
            ),
            detector_configuration=DetectorConfiguration(
                bottom_angle_deg=config.bottom_angle_deg,
                top_angle_deg=config.top_angle_deg,
                hysteresis_deg=config.hysteresis_deg,
                confirmation_frames=config.confirmation_frames,
                min_repetition_duration_s=config.min_repetition_duration_s,
                max_repetition_duration_s=config.max_repetition_duration_s,
                max_measurement_gap_s=config.max_measurement_gap_s,
                smoothing_window_frames=SMOOTHING_WINDOW_FRAMES,
            ),
            accepted_repetition_timestamps=accepted,
            rejected_events=rejected,
        )
    except AssessmentServiceError:
        raise
    except Exception as exc:
        raise AssessmentProcessingError(
            f"Assessment failed ({type(exc).__name__})"
        ) from exc
