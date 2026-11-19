"""Experimental image-plane curl-up measurement and cycle detection.

No curl-up protocol was found in the repository SRS. This implementation
therefore requires explicit angle thresholds and documents its operational
cycle definition in every result; it does not claim to implement a
standardized curl-up test.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from ..pipeline import PoseObservation
from ..video import VideoMetadata
from .base import AssessmentMeasurement, AssessmentResult


# MediaPipe Pose's documented 33-landmark order.
_LANDMARK_INDEX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
}
_SIDES = ("left", "right")


@dataclass(frozen=True)
class CurlUpConfig:
    """Explicit image, signal, and temporal settings for a curl-up experiment.

    ``extended_angle_deg`` and ``flexed_angle_deg`` are included shoulder-hip-
    knee angles, not hip-flexion angles. Since no thresholds or duration are
    specified in the project SRS, callers must provide the angle thresholds;
    ``test_duration_s=None`` means analyze the entire supplied clip.
    """

    image_aspect_ratio: float
    extended_angle_deg: float
    flexed_angle_deg: float
    hysteresis_deg: float
    confirmation_frames: int
    min_repetition_duration_s: float
    max_repetition_duration_s: float
    max_measurement_gap_s: float
    min_visibility: float
    starting_position: Literal["extended"]
    selected_side: Literal["left", "right", "auto"]
    test_duration_s: float | None

    def validate(self) -> None:
        """Reject ambiguous or mathematically unusable settings."""
        if not math.isfinite(self.image_aspect_ratio) or self.image_aspect_ratio <= 0:
            raise ValueError("image_aspect_ratio must be positive and finite")
        if not (
            math.isfinite(self.flexed_angle_deg)
            and math.isfinite(self.extended_angle_deg)
            and 0 <= self.flexed_angle_deg < self.extended_angle_deg <= 180
        ):
            raise ValueError(
                "Require 0 <= flexed angle < extended angle <= 180 degrees"
            )
        if not math.isfinite(self.hysteresis_deg) or self.hysteresis_deg < 0:
            raise ValueError("hysteresis_deg must be finite and non-negative")
        if (
            self.flexed_angle_deg + self.hysteresis_deg
            >= self.extended_angle_deg - self.hysteresis_deg
        ):
            raise ValueError("Hysteresis bands overlap")
        if self.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least 1")
        if (
            not math.isfinite(self.min_repetition_duration_s)
            or not math.isfinite(self.max_repetition_duration_s)
            or self.min_repetition_duration_s < 0
            or self.max_repetition_duration_s <= self.min_repetition_duration_s
        ):
            raise ValueError("Repetition durations must satisfy 0 <= min < max")
        if (
            not math.isfinite(self.max_measurement_gap_s)
            or self.max_measurement_gap_s <= 0
        ):
            raise ValueError("max_measurement_gap_s must be positive and finite")
        if not math.isfinite(self.min_visibility) or not 0 <= self.min_visibility <= 1:
            raise ValueError("min_visibility must be between 0 and 1")
        if self.starting_position != "extended":
            raise ValueError("Only the explicit extended starting position is supported")
        if self.selected_side not in {"left", "right", "auto"}:
            raise ValueError("selected_side must be left, right, or auto")
        if self.test_duration_s is not None and (
            not math.isfinite(self.test_duration_s) or self.test_duration_s <= 0
        ):
            raise ValueError("test_duration_s must be positive and finite when supplied")


def _point(landmark: Any, aspect_ratio: float) -> tuple[float, float] | None:
    """Convert normalized image coordinates to aspect-corrected image-plane xy.

    Multiplying normalized x by width/height puts both axes in units of image
    height. A single uniform scale does not change the resulting angle.
    """
    try:
        x = float(landmark.x) * aspect_ratio
        y = float(landmark.y)
    except (AttributeError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _side_angle(
    landmarks: list[Any], side: str, config: CurlUpConfig
) -> tuple[float | None, float | None, list[str]]:
    """Measure the included shoulder-hip-knee angle on one side."""
    missing: list[str] = []
    joints: dict[str, Any | None] = {}
    visibility_scores: list[float] = []
    points: dict[str, tuple[float, float]] = {}
    for joint in ("shoulder", "hip", "knee"):
        name = f"{side}_{joint}"
        index = _LANDMARK_INDEX[name]
        landmark = landmarks[index] if index < len(landmarks) else None
        joints[joint] = landmark
        if landmark is None:
            missing.append(name)
            continue
        point = _point(landmark, config.image_aspect_ratio)
        if point is None:
            missing.append(name)
            continue
        points[joint] = point
        try:
            visibility = float(landmark.visibility)
        except (AttributeError, TypeError, ValueError):
            visibility = math.nan
        if not math.isfinite(visibility):
            missing.append(f"{name}_visibility")
            continue
        visibility_scores.append(visibility)
        if visibility < config.min_visibility:
            missing.append(f"{name}_low_visibility")
        presence = getattr(landmark, "presence", None)
        if presence is not None:
            try:
                presence_value = float(presence)
            except (TypeError, ValueError):
                presence_value = math.nan
            if not math.isfinite(presence_value) or presence_value < config.min_visibility:
                missing.append(f"{name}_low_presence")

    if missing or len(points) != 3 or len(visibility_scores) != 3:
        return None, min(visibility_scores) if visibility_scores else None, missing

    shoulder, hip, knee = points["shoulder"], points["hip"], points["knee"]
    trunk = (shoulder[0] - hip[0], shoulder[1] - hip[1])
    thigh = (knee[0] - hip[0], knee[1] - hip[1])
    trunk_length = math.hypot(*trunk)
    thigh_length = math.hypot(*thigh)
    if trunk_length == 0 or thigh_length == 0:
        return None, min(visibility_scores), [f"{side}_degenerate_geometry"]
    cosine = (trunk[0] * thigh[0] + trunk[1] * thigh[1]) / (
        trunk_length * thigh_length
    )
    included_angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return included_angle, min(visibility_scores), []


def measure_curl_up_angle(
    landmarks: list[Any], config: CurlUpConfig
) -> tuple[float | None, str, float | None, dict[str, float | None], list[str]]:
    """Return selected included angle, side, confidence, per-side angles, gaps."""
    config.validate()
    side_results = {
        side: _side_angle(landmarks, side, config) for side in _SIDES
    }
    per_side = {side: side_results[side][0] for side in _SIDES}
    eligible = [
        (side, side_results[side][0], side_results[side][1])
        for side in _SIDES
        if side_results[side][0] is not None
        and config.selected_side in {side, "auto"}
    ]
    if not eligible:
        missing = [
            item
            for side in _SIDES
            if config.selected_side in {side, "auto"}
            for item in side_results[side][2]
        ]
        return None, "", None, per_side, missing
    side, angle, confidence = max(
        eligible,
        key=lambda item: (
            item[2] if item[2] is not None else -1.0,
            item[0] == "left",
        ),
    )
    return angle, side, confidence, per_side, []


class CurlUpPhaseDetector:
    """Count configured extended → flexed → extended cycles only.

    This is an experimental signal state machine, not a standardized test
    definition. It requires a confirmed extended posture before accepting a
    cycle and rejects cycles interrupted by an excessive measurement gap.
    """

    SEEK_EXTENDED = "SEEK_EXTENDED"
    EXTENDED = "EXTENDED"
    FLEXING = "FLEXING"
    FLEXED = "FLEXED"
    RETURNING = "RETURNING"

    def __init__(self, config: CurlUpConfig):
        config.validate()
        self.config = config
        self.state = self.SEEK_EXTENDED
        self.repetitions = 0
        self.events: list[dict[str, Any]] = []
        self._last_valid_timestamp_ms: int | None = None
        self._cycle_start_ms: int | None = None
        self._partial_start_ms: int | None = None
        self._has_seen_extended = False
        self._confirmation_name: str | None = None
        self._confirmation_count = 0
        self._confirmation_start_ms: int | None = None

    def _clear_confirmation(self) -> None:
        self._confirmation_name = None
        self._confirmation_count = 0
        self._confirmation_start_ms = None

    def _confirm(self, name: str, condition: bool, timestamp_ms: int) -> int | None:
        if not condition:
            self._clear_confirmation()
            return None
        if self._confirmation_name == name:
            self._confirmation_count += 1
        else:
            self._confirmation_name = name
            self._confirmation_count = 1
            self._confirmation_start_ms = timestamp_ms
        if self._confirmation_count < self.config.confirmation_frames:
            return None
        confirmed_at = self._confirmation_start_ms
        self._clear_confirmation()
        return confirmed_at

    def _emit_rejected(self, reason: str, end_timestamp_ms: int) -> None:
        start = self._cycle_start_ms
        if start is None:
            start = self._partial_start_ms
        self.events.append(
            {
                "event_type": "rejected",
                "repetition_number": None,
                "start_timestamp_ms": start,
                "end_timestamp_ms": end_timestamp_ms,
                "duration_seconds": (
                    round((end_timestamp_ms - start) / 1000.0, 3)
                    if start is not None
                    else None
                ),
                "reason": reason,
            }
        )

    def _reset(self, state: str = SEEK_EXTENDED) -> None:
        self.state = state
        self._cycle_start_ms = None
        self._partial_start_ms = None
        self._clear_confirmation()

    def _reject_cycle(self, reason: str, timestamp_ms: int) -> None:
        self._emit_rejected(reason, timestamp_ms)
        self._reset()

    def _check_gap(self, timestamp_ms: int) -> bool:
        if self._last_valid_timestamp_ms is None:
            return False
        gap_s = (timestamp_ms - self._last_valid_timestamp_ms) / 1000.0
        if gap_s <= self.config.max_measurement_gap_s:
            return False
        if self.state in {self.FLEXING, self.FLEXED, self.RETURNING}:
            self._reject_cycle("long_measurement_gap", timestamp_ms)
        elif self._partial_start_ms is not None:
            self._emit_rejected("long_measurement_gap", timestamp_ms)
            self._reset()
        else:
            self._reset()
        return True

    def _finish_cycle(self, end_timestamp_ms: int) -> None:
        start = self._cycle_start_ms
        if start is None:
            self._reject_cycle("missing_cycle_start", end_timestamp_ms)
            return
        duration_s = (end_timestamp_ms - start) / 1000.0
        if duration_s < self.config.min_repetition_duration_s:
            self._reject_cycle("too_short", end_timestamp_ms)
            return
        if duration_s > self.config.max_repetition_duration_s:
            self._reject_cycle("excessively_long", end_timestamp_ms)
            return
        self.repetitions += 1
        self.events.append(
            {
                "event_type": "accepted",
                "repetition_number": self.repetitions,
                "start_timestamp_ms": start,
                "end_timestamp_ms": end_timestamp_ms,
                "duration_seconds": round(duration_s, 3),
                "reason": None,
            }
        )
        self.state = self.EXTENDED
        self._cycle_start_ms = None
        self._clear_confirmation()

    def update(
        self, frame_number: int, timestamp_ms: int, angle_deg: float | None
    ) -> dict[str, Any]:
        """Consume one included angle and return a frame-level state record."""
        event_start = len(self.events)
        if angle_deg is not None and not math.isfinite(angle_deg):
            angle_deg = None

        if angle_deg is None:
            self._clear_confirmation()
            if self._last_valid_timestamp_ms is not None:
                gap_s = (timestamp_ms - self._last_valid_timestamp_ms) / 1000.0
                if gap_s > self.config.max_measurement_gap_s:
                    if self.state in {self.FLEXING, self.FLEXED, self.RETURNING}:
                        self._reject_cycle("long_measurement_gap", timestamp_ms)
                    elif self._partial_start_ms is not None:
                        self._emit_rejected("long_measurement_gap", timestamp_ms)
                        self._reset()
                    else:
                        self._reset()
            return self._frame_result(frame_number, timestamp_ms, None, event_start)

        self._check_gap(timestamp_ms)
        self._last_valid_timestamp_ms = timestamp_ms
        if (
            self._cycle_start_ms is not None
            and (timestamp_ms - self._cycle_start_ms) / 1000.0
            > self.config.max_repetition_duration_s
        ):
            self._reject_cycle("excessively_long", timestamp_ms)

        extended = self.config.extended_angle_deg
        flexed = self.config.flexed_angle_deg
        hysteresis = self.config.hysteresis_deg

        if self.state == self.SEEK_EXTENDED:
            confirmed = self._confirm(
                "initial_extended", angle_deg >= extended, timestamp_ms
            )
            if angle_deg <= flexed and self._partial_start_ms is None:
                self._partial_start_ms = timestamp_ms
            if confirmed is not None:
                if self._partial_start_ms is not None:
                    self._emit_rejected(
                        "video_started_mid_repetition", confirmed
                    )
                self._has_seen_extended = True
                self._reset(self.EXTENDED)

        elif self.state == self.EXTENDED:
            left = self._confirm(
                "leave_extended", angle_deg <= extended - hysteresis, timestamp_ms
            )
            if left is not None:
                self.state = self.FLEXING
                self._cycle_start_ms = left

        elif self.state == self.FLEXING:
            if angle_deg <= flexed:
                reached = self._confirm("reach_flexed", True, timestamp_ms)
                if reached is not None:
                    self.state = self.FLEXED
            elif angle_deg >= extended:
                aborted = self._confirm("abort_before_flexion", True, timestamp_ms)
                if aborted is not None:
                    self._emit_rejected("flexion_not_reached", aborted)
                    self._reset(self.EXTENDED)
            else:
                self._clear_confirmation()

        elif self.state == self.FLEXED:
            left = self._confirm(
                "leave_flexed", angle_deg >= flexed + hysteresis, timestamp_ms
            )
            if left is not None:
                self.state = self.RETURNING

        elif self.state == self.RETURNING:
            if angle_deg >= extended:
                reached = self._confirm("return_extended", True, timestamp_ms)
                if reached is not None:
                    self._finish_cycle(reached)
            elif angle_deg <= flexed:
                returned = self._confirm("return_to_flexed", True, timestamp_ms)
                if returned is not None:
                    self.state = self.FLEXED
            else:
                self._clear_confirmation()

        return self._frame_result(frame_number, timestamp_ms, angle_deg, event_start)

    def _frame_result(
        self,
        frame_number: int,
        timestamp_ms: int,
        angle_deg: float | None,
        event_start: int,
    ) -> dict[str, Any]:
        return {
            "frame_number": frame_number,
            "timestamp_ms": timestamp_ms,
            "included_angle_deg": angle_deg,
            "movement_state": self.state,
            "repetition_count": self.repetitions,
            "events": self.events[event_start:],
        }

    def finish(self, timestamp_ms: int) -> None:
        """Reject any uncompleted cycle at clip or configured-test end."""
        if self.state in {self.FLEXING, self.FLEXED, self.RETURNING}:
            self._reject_cycle("end_of_test_incomplete_cycle", timestamp_ms)
        elif self.state == self.SEEK_EXTENDED and self._partial_start_ms is not None:
            self._emit_rejected("video_started_mid_repetition", timestamp_ms)
            self._reset()


class CurlUpAssessment:
    """Measure and count configured image-plane curl-up cycles.

    The operational repetition definition is: begin from a confirmed extended
    shoulder-hip-knee posture, cross the configured flexed-angle threshold, and
    return across the configured extended-angle threshold. Protocol details
    absent from the SRS remain caller-supplied or explicitly unresolved.
    """

    def __init__(self, config: CurlUpConfig):
        config.validate()
        self.config = config

    def measure_frame(self, observation: PoseObservation) -> AssessmentMeasurement:
        """Measure included shoulder-hip-knee angle(s) for one pose frame."""
        pose_detected = observation.pose_detected
        landmarks = observation.pose_landmarks[0] if pose_detected else []
        angle, side, visibility, per_side, missing = measure_curl_up_angle(
            landmarks, self.config
        )
        return AssessmentMeasurement(
            frame_number=observation.frame_number,
            timestamp_ms=observation.timestamp_ms,
            values={
                "included_angle_deg": angle,
                "curl_flexion_deg": 180.0 - angle if angle is not None else None,
                "selected_side": side,
                "left_included_angle_deg": per_side["left"],
                "right_included_angle_deg": per_side["right"],
            },
            quality={
                "pose_detected": pose_detected,
                "measurement_valid": angle is not None,
                "visibility_score": visibility,
                "missing_landmarks": ",".join(missing),
            },
        )

    def analyze(
        self,
        measurements: Sequence[AssessmentMeasurement],
        video: VideoMetadata,
    ) -> AssessmentResult:
        """Run the independent phase detector and return measurements/events."""
        detector = CurlUpPhaseDetector(self.config)
        if self.config.test_duration_s is None:
            selected = list(measurements)
        else:
            duration_ms = self.config.test_duration_s * 1000.0
            selected = [item for item in measurements if item.timestamp_ms <= duration_ms]

        frame_rows: list[dict[str, Any]] = []
        pose_frames = 0
        valid_angles: list[float] = []
        for item in selected:
            pose_detected = bool(item.quality.get("pose_detected", False))
            pose_frames += int(pose_detected)
            angle = item.values.get("included_angle_deg")
            if angle is not None:
                angle = float(angle)
                if math.isfinite(angle):
                    valid_angles.append(angle)
                else:
                    angle = None
            frame_state = detector.update(
                item.frame_number, item.timestamp_ms, angle
            )
            frame_rows.append(
                {
                    "frame_number": item.frame_number,
                    "timestamp_ms": item.timestamp_ms,
                    "included_angle_deg": angle,
                    "curl_flexion_deg": (
                        180.0 - angle if angle is not None else None
                    ),
                    "selected_side": item.values.get("selected_side", ""),
                    "movement_state": frame_state["movement_state"],
                    "repetition_count": frame_state["repetition_count"],
                    "measurement_valid": bool(
                        item.quality.get("measurement_valid", angle is not None)
                    ),
                    "visibility_score": item.quality.get("visibility_score"),
                    "missing_landmarks": item.quality.get("missing_landmarks", ""),
                }
            )

        if selected:
            final_timestamp = selected[-1].timestamp_ms
            if len(selected) > 1:
                final_timestamp += max(
                    0, selected[-1].timestamp_ms - selected[-2].timestamp_ms
                )
            detector.finish(final_timestamp)

        frame_count = len(selected)
        accepted = [
            event for event in detector.events if event["event_type"] == "accepted"
        ]
        return AssessmentResult(
            assessment_type="curl_up",
            summary={
                "detected_repetitions": detector.repetitions,
                "total_frames_analyzed": frame_count,
                "frames_with_detected_poses": pose_frames,
                "frames_with_valid_measurements": len(valid_angles),
                "pose_detection_percentage": (
                    round(pose_frames * 100.0 / frame_count, 2)
                    if frame_count
                    else 0.0
                ),
                "measurement_coverage_percentage": (
                    round(len(valid_angles) * 100.0 / frame_count, 2)
                    if frame_count
                    else 0.0
                ),
                "average_included_angle_deg": (
                    round(sum(valid_angles) / len(valid_angles), 3)
                    if valid_angles
                    else None
                ),
                "protocol_status": "experimental_protocol_not_defined_in_srs",
                "protocol_gaps": [
                    "The SRS does not specify a curl-up repetition standard or angle criteria.",
                    "The SRS does not specify a starting posture; this detector requires an extended posture.",
                    "The SRS does not specify a test duration; the full clip is analyzed unless configured.",
                ],
                "measurements": frame_rows,
                "accepted_repetitions": accepted,
            },
            events=tuple(detector.events),
            configuration=asdict(self.config),
        )

