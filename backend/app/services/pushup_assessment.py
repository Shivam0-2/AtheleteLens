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
from athletelens_ai.signals import rolling_median
from athletelens_ai.video import (
    VideoMetadata as EngineVideoMetadata,
    frame_timestamp_ms,
    iter_video_frames,
    open_video,
)

from app.schemas.assessment import (
    DetectorConfiguration,
    PushUpAssessmentResponse,
    RejectedRepetition,
    RepetitionTimestamp,
    VideoMetadata,
)


MIN_LANDMARK_VISIBILITY = 0.5
SMOOTHING_WINDOW_FRAMES = 7


class AssessmentServiceError(Exception):
    """Base error for failures that can be reported by the upload endpoint."""


class InvalidVideoError(AssessmentServiceError):
    """The submitted path does not contain a readable video stream."""


class MissingPoseModelError(AssessmentServiceError):
    """The configured MediaPipe task model is unavailable."""


class AssessmentProcessingError(AssessmentServiceError):
    """Pose inference or measurement processing failed."""


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

    capture = None
    try:
        cv2, mp, np = load_pose_dependencies()
        try:
            capture, metadata = open_video(video_path, cv2)
        except (OSError, ValueError) as exc:
            raise InvalidVideoError("Uploaded file is not a readable video") from exc

        if (
            not math.isfinite(metadata.fps)
            or metadata.fps <= 0
            or metadata.width <= 0
            or metadata.height <= 0
        ):
            raise InvalidVideoError("Video metadata is invalid")

        timestamps: list[int] = []
        raw_angles: list[float | None] = []
        frames_with_poses = 0
        previous_timestamp_ms = -1
        try:
            landmarker_context = create_video_landmarker(mp, str(model_path))
            with landmarker_context as landmarker:
                for frame_index, frame_bgr in iter_video_frames(capture):
                    timestamp_ms = frame_timestamp_ms(
                        frame_index, metadata.fps, previous_timestamp_ms
                    )
                    previous_timestamp_ms = timestamp_ms
                    result = process_video_frame(
                        frame_bgr, timestamp_ms, landmarker, cv2, mp, np
                    )
                    pose_landmarks = result.pose_landmarks or []
                    if pose_landmarks:
                        frames_with_poses += 1
                        angle, _side, _left, _right, _fields = choose_elbow_angle(
                            pose_landmarks[0], MIN_LANDMARK_VISIBILITY
                        )
                        if angle is not None and not math.isfinite(angle):
                            angle = None
                    else:
                        angle = None
                    timestamps.append(timestamp_ms)
                    raw_angles.append(angle)
        except AssessmentServiceError:
            raise
        except Exception as exc:
            raise AssessmentProcessingError(
                f"Pose processing failed ({type(exc).__name__})"
            ) from exc
        finally:
            capture.release()
            capture = None

        if not timestamps:
            raise InvalidVideoError("Uploaded video contains no readable frames")

        smoothed_angles = rolling_median(raw_angles, SMOOTHING_WINDOW_FRAMES)
        detector = PhaseBasedPushUpDetector(config)
        for index, (timestamp_ms, angle) in enumerate(
            zip(timestamps, smoothed_angles), start=1
        ):
            detector.update(index, timestamp_ms, angle)
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
        return PushUpAssessmentResponse(
            input_filename=input_filename,
            video=VideoMetadata(
                fps=metadata.fps, width=metadata.width, height=metadata.height
            ),
            total_frames_processed=frames_processed,
            frames_with_detected_poses=frames_with_poses,
            frames_with_valid_elbow_angle=len(valid_angles),
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
    finally:
        if capture is not None:
            capture.release()
