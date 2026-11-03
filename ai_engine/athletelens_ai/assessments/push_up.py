"""Reusable deterministic push-up measurements and repetition detectors.

The baseline threshold counter and experimental phase detector remain distinct.
Neither detector has been validated for sports assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
JOINT_NAMES = ("shoulder", "elbow", "wrist")
LANDMARK_FIELDS = ("x", "y", "z", "visibility", "presence")


@dataclass
class PushUpRepetitionCounter:
    """Count a repetition after a lower-threshold then upper-threshold crossing."""

    lower_angle_deg: float
    upper_angle_deg: float
    repetitions: int = 0
    phase: str = "seeking_bottom"

    def update(self, elbow_angle_deg: float | None) -> int:
        """Update the state from one angle; missing angles do not change state."""
        if elbow_angle_deg is None:
            return self.repetitions
        if self.phase == "seeking_bottom" and elbow_angle_deg <= self.lower_angle_deg:
            self.phase = "bottom_reached"
        elif self.phase == "bottom_reached" and elbow_angle_deg >= self.upper_angle_deg:
            self.repetitions += 1
            self.phase = "top_reached"
        elif self.phase == "top_reached" and elbow_angle_deg <= self.lower_angle_deg:
            self.phase = "bottom_reached"
        return self.repetitions


def elbow_angle_degrees(shoulder: Any, elbow: Any, wrist: Any) -> float | None:
    """Compute the 2D shoulder-elbow-wrist angle in degrees, or None if invalid."""
    try:
        shoulder_xy = (float(shoulder.x), float(shoulder.y))
        elbow_xy = (float(elbow.x), float(elbow.y))
        wrist_xy = (float(wrist.x), float(wrist.y))
    except (AttributeError, TypeError, ValueError):
        return None

    upper_arm = (shoulder_xy[0] - elbow_xy[0], shoulder_xy[1] - elbow_xy[1])
    forearm = (wrist_xy[0] - elbow_xy[0], wrist_xy[1] - elbow_xy[1])
    upper_length = math.hypot(*upper_arm)
    forearm_length = math.hypot(*forearm)
    if upper_length == 0 or forearm_length == 0:
        return None

    cosine = (
        upper_arm[0] * forearm[0] + upper_arm[1] * forearm[1]
    ) / (upper_length * forearm_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def landmark_row_fields(prefix: str, landmark: Any | None) -> dict[str, float | str]:
    """Return supplied landmark fields for one joint, blank if unavailable."""
    values: dict[str, float | str] = {}
    for field in LANDMARK_FIELDS:
        value = getattr(landmark, field, None) if landmark is not None else None
        values[f"{prefix}_{field}"] = float(value) if value is not None else ""
    return values


def side_measurement(
    landmarks: list[Any], side: str, min_visibility: float
) -> tuple[float | None, dict[str, float | str], float | None]:
    """Extract a side's shoulder/elbow/wrist and compute its visible angle."""
    import mediapipe as mp

    pose_landmark = mp.tasks.vision.PoseLandmark
    indices = {
        "shoulder": getattr(pose_landmark, f"{side.upper()}_SHOULDER").value,
        "elbow": getattr(pose_landmark, f"{side.upper()}_ELBOW").value,
        "wrist": getattr(pose_landmark, f"{side.upper()}_WRIST").value,
    }
    joints: dict[str, Any | None] = {
        joint: landmarks[index] if index < len(landmarks) else None
        for joint, index in indices.items()
    }
    fields: dict[str, float | str] = {}
    for joint, landmark in joints.items():
        fields.update(landmark_row_fields(f"{side}_{joint}", landmark))

    confidences = [
        float(landmark.visibility)
        for landmark in joints.values()
        if landmark is not None and getattr(landmark, "visibility", None) is not None
    ]
    visibility_score = min(confidences) if len(confidences) == len(JOINT_NAMES) else None
    visibility_ok = all(
        landmark is not None
        and (
            getattr(landmark, "visibility", None) is None
            or float(landmark.visibility) >= min_visibility
        )
        for landmark in joints.values()
    )
    angle = (
        elbow_angle_degrees(joints["shoulder"], joints["elbow"], joints["wrist"])
        if visibility_ok
        else None
    )
    return angle, fields, visibility_score


def choose_elbow_angle(
    landmarks: list[Any], min_visibility: float
) -> tuple[float | None, str, float | None, float | None, dict[str, float | str]]:
    """Choose the valid side with strongest visibility; prefer left on ties."""
    left_angle, left_fields, left_score = side_measurement(
        landmarks, "left", min_visibility
    )
    right_angle, right_fields, right_score = side_measurement(
        landmarks, "right", min_visibility
    )
    fields = {**left_fields, **right_fields}

    candidates = [
        ("left", left_angle, left_score),
        ("right", right_angle, right_score),
    ]
    candidates = [candidate for candidate in candidates if candidate[1] is not None]
    if not candidates:
        return None, "", left_angle, right_angle, fields
    side, angle, _ = max(
        candidates,
        key=lambda candidate: (
            candidate[2] if candidate[2] is not None else -1.0,
            candidate[0] == "left",
        ),
    )
    return angle, side, left_angle, right_angle, fields


@dataclass(frozen=True)
class DetectorConfig:
    """Configurable phase thresholds, confirmation, duration, and gap limits."""

    bottom_angle_deg: float = 90.0
    top_angle_deg: float = 160.0
    hysteresis_deg: float = 5.0
    confirmation_frames: int = 2
    min_repetition_duration_s: float = 0.25
    max_repetition_duration_s: float = 3.0
    max_measurement_gap_s: float = 0.25

    def validate(self) -> None:
        if not (0 <= self.bottom_angle_deg < self.top_angle_deg <= 180):
            raise ValueError("Require 0 <= bottom angle < top angle <= 180 degrees")
        if not math.isfinite(self.hysteresis_deg) or self.hysteresis_deg < 0:
            raise ValueError("Hysteresis must be a finite non-negative angle")
        if self.bottom_angle_deg + self.hysteresis_deg >= self.top_angle_deg - self.hysteresis_deg:
            raise ValueError("Hysteresis bands overlap; widen the threshold gap or reduce hysteresis")
        if self.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least 1")
        if (
            not math.isfinite(self.min_repetition_duration_s)
            or not math.isfinite(self.max_repetition_duration_s)
            or self.min_repetition_duration_s < 0
            or self.max_repetition_duration_s <= self.min_repetition_duration_s
        ):
            raise ValueError("Repetition durations must satisfy 0 <= min < max")
        if not math.isfinite(self.max_measurement_gap_s) or self.max_measurement_gap_s <= 0:
            raise ValueError("max_measurement_gap_s must be positive and finite")

class PhaseBasedPushUpDetector:
    """Count only observed TOP → DESCENDING → BOTTOM → ASCENDING → TOP cycles."""

    SEEK_TOP = "SEEK_TOP"
    TOP = "TOP"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"

    def __init__(self, config: DetectorConfig):
        config.validate()
        self.config = config
        self.state = self.SEEK_TOP
        self.repetitions = 0
        self.events: list[dict[str, Any]] = []
        self._last_valid_timestamp_ms: int | None = None
        self._cycle_start_ms: int | None = None
        self._seek_partial_start_ms: int | None = None
        self._has_seen_top = False
        self._confirmation_name: str | None = None
        self._confirmation_count = 0
        self._confirmation_start_ms: int | None = None

    def _clear_confirmation(self) -> None:
        self._confirmation_name = None
        self._confirmation_count = 0
        self._confirmation_start_ms = None

    def _confirm(self, name: str, condition: bool, timestamp_ms: int) -> int | None:
        """Return the first timestamp once a condition holds for N valid frames."""
        if not condition:
            self._clear_confirmation()
            return None
        if self._confirmation_name == name:
            self._confirmation_count += 1
        else:
            self._confirmation_name = name
            self._confirmation_count = 1
            self._confirmation_start_ms = timestamp_ms
        if self._confirmation_count >= self.config.confirmation_frames:
            first_timestamp = self._confirmation_start_ms
            self._clear_confirmation()
            return first_timestamp
        return None

    def _emit_rejection(self, reason: str, end_timestamp_ms: int) -> None:
        start = self._cycle_start_ms
        if start is None:
            start = self._seek_partial_start_ms
        self.events.append(
            {
                "event_type": "rejected",
                "repetition_number": "",
                "start_timestamp_ms": start if start is not None else "",
                "end_timestamp_ms": end_timestamp_ms,
                "duration_seconds": (
                    round((end_timestamp_ms - start) / 1000.0, 3)
                    if start is not None
                    else ""
                ),
                "reason": reason,
            }
        )

    def _reset_to_seek_top(self) -> None:
        self.state = self.SEEK_TOP
        self._cycle_start_ms = None
        self._seek_partial_start_ms = None
        self._clear_confirmation()

    def _reject_cycle(self, reason: str, end_timestamp_ms: int) -> None:
        self._emit_rejection(reason, end_timestamp_ms)
        self._reset_to_seek_top()

    def _long_gap(self, timestamp_ms: int) -> bool:
        if self._last_valid_timestamp_ms is None:
            return False
        elapsed_s = (timestamp_ms - self._last_valid_timestamp_ms) / 1000.0
        if elapsed_s <= self.config.max_measurement_gap_s:
            return False
        if self.state in {self.DESCENDING, self.BOTTOM, self.ASCENDING}:
            self._reject_cycle("long_measurement_gap", timestamp_ms)
        elif self._seek_partial_start_ms is not None:
            self._emit_rejection("long_measurement_gap", timestamp_ms)
            self._reset_to_seek_top()
        else:
            self._reset_to_seek_top()
        return True

    def _record_completion(self, start_ms: int, end_ms: int) -> None:
        duration_s = (end_ms - start_ms) / 1000.0
        if duration_s < self.config.min_repetition_duration_s:
            self._reject_cycle("too_short", end_ms)
            return
        if duration_s > self.config.max_repetition_duration_s:
            self._reject_cycle("excessively_long", end_ms)
            return
        self.repetitions += 1
        self.events.append(
            {
                "event_type": "completed",
                "repetition_number": self.repetitions,
                "start_timestamp_ms": start_ms,
                "end_timestamp_ms": end_ms,
                "duration_seconds": round(duration_s, 3),
                "reason": "",
            }
        )
        self.state = self.TOP
        self._cycle_start_ms = None
        self._clear_confirmation()

    def update(
        self, frame_number: int, timestamp_ms: int, angle_deg: float | None
    ) -> dict[str, Any]:
        """Consume one frame and return its resulting movement state."""
        event_start = len(self.events)
        if angle_deg is not None and not math.isfinite(angle_deg):
            angle_deg = None

        if angle_deg is None:
            self._clear_confirmation()
            if self._last_valid_timestamp_ms is not None:
                gap_s = (timestamp_ms - self._last_valid_timestamp_ms) / 1000.0
                if gap_s > self.config.max_measurement_gap_s:
                    if self.state in {self.DESCENDING, self.BOTTOM, self.ASCENDING}:
                        self._reject_cycle("long_measurement_gap", timestamp_ms)
                    elif self._seek_partial_start_ms is not None:
                        self._emit_rejection("long_measurement_gap", timestamp_ms)
                        self._reset_to_seek_top()
                    else:
                        self._reset_to_seek_top()
            return self._frame_result(frame_number, timestamp_ms, None, event_start)

        self._long_gap(timestamp_ms)
        self._last_valid_timestamp_ms = timestamp_ms

        if (
            self._cycle_start_ms is not None
            and (timestamp_ms - self._cycle_start_ms) / 1000.0
            > self.config.max_repetition_duration_s
        ):
            self._reject_cycle("excessively_long", timestamp_ms)

        bottom = self.config.bottom_angle_deg
        bottom_exit = bottom + self.config.hysteresis_deg
        top = self.config.top_angle_deg
        top_exit = top - self.config.hysteresis_deg

        if self.state == self.SEEK_TOP:
            confirmed = self._confirm("top", angle_deg >= top, timestamp_ms)
            if angle_deg <= bottom and self._seek_partial_start_ms is None:
                self._seek_partial_start_ms = timestamp_ms
            if confirmed is not None:
                if self._seek_partial_start_ms is not None:
                    self.events.append(
                        {
                            "event_type": "rejected",
                            "repetition_number": "",
                            "start_timestamp_ms": self._seek_partial_start_ms,
                            "end_timestamp_ms": confirmed,
                            "duration_seconds": round(
                                (confirmed - self._seek_partial_start_ms) / 1000.0, 3
                            ),
                            "reason": (
                                "video_started_mid_repetition"
                                if not self._has_seen_top
                                else "no_confirmed_top_before_descent"
                            ),
                        }
                    )
                self.state = self.TOP
                self._seek_partial_start_ms = None
                self._has_seen_top = True

        elif self.state == self.TOP:
            confirmed = self._confirm("leave_top", angle_deg <= top_exit, timestamp_ms)
            if confirmed is not None:
                self.state = self.DESCENDING
                self._cycle_start_ms = confirmed

        elif self.state == self.DESCENDING:
            if angle_deg <= bottom:
                confirmed = self._confirm("reach_bottom", True, timestamp_ms)
                if confirmed is not None:
                    self.state = self.BOTTOM
            elif angle_deg >= top:
                recovered = self._confirm("abort_descent", True, timestamp_ms)
                if recovered is not None:
                    self._reject_cycle("bottom_not_reached", recovered)
            else:
                self._clear_confirmation()

        elif self.state == self.BOTTOM:
            confirmed = self._confirm("leave_bottom", angle_deg >= bottom_exit, timestamp_ms)
            if confirmed is not None:
                self.state = self.ASCENDING

        elif self.state == self.ASCENDING:
            if angle_deg >= top:
                reached_top = self._confirm("reach_top", True, timestamp_ms)
                if reached_top is not None:
                    start_ms = (
                        self._cycle_start_ms
                        if self._cycle_start_ms is not None
                        else reached_top
                    )
                    self._record_completion(start_ms, reached_top)
            elif angle_deg <= bottom:
                returned_to_bottom = self._confirm(
                    "return_to_bottom", True, timestamp_ms
                )
                if returned_to_bottom is not None:
                    self.state = self.BOTTOM
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
            "angle_deg": angle_deg if angle_deg is not None else "",
            "movement_state": self.state,
            "repetition_count": self.repetitions,
            "events": self.events[event_start:],
        }

    def finish(self, timestamp_ms: int) -> None:
        """Record an incomplete cycle at end-of-video without counting it."""
        if self.state in {self.DESCENDING, self.BOTTOM, self.ASCENDING}:
            self._reject_cycle("end_of_video_incomplete_cycle", timestamp_ms)
        elif self.state == self.SEEK_TOP and self._seek_partial_start_ms is not None:
            reason = (
                "video_started_mid_repetition"
                if not self._has_seen_top
                else "no_confirmed_top_before_descent"
            )
            self._emit_rejection(reason, timestamp_ms)
            self._reset_to_seek_top()

