"""Experimental side-view standing vertical-jump assessment.

The current height field is explicitly flight-time-derived: it estimates
takeoff-to-apex flight height as ``g * t**2 / 8`` under equal takeoff/landing
center-of-mass height. It is not a pose-derived center-of-mass displacement.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Sequence

from .base import AssessmentMeasurement, AssessmentProtocol, AssessmentResult
from ..pipeline import PoseObservation
from ..video import VideoMetadata

# MediaPipe PoseLandmarker indices: ankles, heels, foot indices, and hips.
_FOOT_IDS = (27, 28, 29, 30, 31, 32)
_HIP_IDS = (23, 24)


@dataclass(frozen=True)
class VerticalJumpConfig:
    gravity_m_s2: float = 9.80665
    min_visibility: float = 0.5
    baseline_confirmation_frames: int = 5
    baseline_max_variation: float = 0.015
    attempt_initialization_displacement: float = 0.01
    takeoff_displacement: float = 0.025
    landing_displacement: float = 0.012
    confirmation_frames: int = 2
    max_measurement_gap_s: float = 0.25
    min_flight_duration_s: float = 0.12
    max_flight_duration_s: float = 1.5
    min_attempt_duration_s: float = 0.2
    max_attempt_duration_s: float = 5.0

    def validate(self) -> None:
        if not math.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0:
            raise ValueError("gravity_m_s2 must be positive and finite")
        if not math.isfinite(self.min_visibility) or not 0 <= self.min_visibility <= 1:
            raise ValueError("min_visibility must be between 0 and 1")
        if (isinstance(self.baseline_confirmation_frames, bool) or self.baseline_confirmation_frames < 2
                or isinstance(self.confirmation_frames, bool) or self.confirmation_frames < 1):
            raise ValueError("Confirmation frame counts are too small")
        for name in ("baseline_max_variation", "attempt_initialization_displacement", "takeoff_displacement", "landing_displacement"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.landing_displacement >= self.takeoff_displacement:
            raise ValueError("landing_displacement must be below takeoff_displacement")
        for name in ("max_measurement_gap_s", "min_flight_duration_s", "max_flight_duration_s", "min_attempt_duration_s", "max_attempt_duration_s"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.max_measurement_gap_s <= 0:
            raise ValueError("max_measurement_gap_s must be positive and finite")
        if not (0 < self.min_flight_duration_s < self.max_flight_duration_s):
            raise ValueError("Flight duration limits must satisfy 0 < min < max")
        if not (0 <= self.min_attempt_duration_s < self.max_attempt_duration_s):
            raise ValueError("Attempt duration limits must satisfy 0 <= min < max")


def _visible(landmark: Any, minimum: float) -> bool:
    try:
        visibility = float(landmark.visibility)
        x, y = float(landmark.x), float(landmark.y)
        presence = getattr(landmark, "presence", None)
        presence_ok = presence is None or (math.isfinite(float(presence)) and float(presence) >= minimum)
    except (AttributeError, TypeError, ValueError):
        return False
    return math.isfinite(x) and math.isfinite(y) and math.isfinite(visibility) and visibility >= minimum and presence_ok


class VerticalJumpAssessment(AssessmentProtocol):
    """Measure visible foot/hip motion and analyze baseline→flight→landing cycles."""

    def __init__(self, config: VerticalJumpConfig | None = None):
        self.config = config or VerticalJumpConfig()
        self.config.validate()

    def measure_frame(self, observation: PoseObservation) -> AssessmentMeasurement:
        points = observation.pose_landmarks[0] if observation.pose_detected else []
        feet = [points[i] for i in _FOOT_IDS if i < len(points) and points[i] is not None and _visible(points[i], self.config.min_visibility)]
        hips = [points[i] for i in _HIP_IDS if i < len(points) and points[i] is not None and _visible(points[i], self.config.min_visibility)]
        # Ground proxy: lowest visible foot point in normalized image coordinates.
        foot_y = max((float(point.y) for point in feet), default=None)
        hip_y = sum(float(point.y) for point in hips) / len(hips) if len(hips) == 2 else None
        confidences = [float(point.visibility) for point in feet + hips]
        complete = foot_y is not None and hip_y is not None and len(hips) == 2 and len(feet) >= 2
        return AssessmentMeasurement(
            observation.frame_number, observation.timestamp_ms,
            {"foot_ground_y": foot_y if complete else None, "hip_center_y": hip_y if complete else None},
            {"pose_detected": observation.pose_detected, "measurement_valid": complete,
             "visibility_score": min(confidences) if complete else None,
             "missing_landmarks": "" if complete else "feet_or_bilateral_hips"},
        )

    def analyze(self, measurements: Sequence[AssessmentMeasurement], video: VideoMetadata) -> AssessmentResult:
        """Detect attempt start, confirmed loss of contact, then confirmed landing.

        Contact is a video proxy: both feet are represented by the lowest
        visible ankle/heel/toe y-coordinate. ``*_timestamp_ms`` event fields
        are the frame at which the configured confirmation count completes.
        """
        cfg = self.config
        state = "SEEK_STANDING_BASELINE"
        baseline: list[tuple[float, float]] = []
        baseline_foot = baseline_hip = None
        startup_origin: tuple[float, float, int] | None = None
        startup_rejection_recorded = False
        last_valid_ms: int | None = None
        confirmation_name: str | None = None
        confirmation_count = 0
        confirmation_start_ms: int | None = None
        attempt_start_ms: int | None = None
        takeoff_candidate_ms: int | None = None
        takeoff_confirmation_ms: int | None = None
        landing_candidate_ms: int | None = None
        events: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        quality_scores: list[float] = []
        attempt_quality: list[float] = []

        active_states = {"ATTEMPT_INITIALIZED", "TAKEOFF_CANDIDATE", "FLIGHT", "LANDING_CANDIDATE"}

        def clear_confirmation() -> None:
            nonlocal confirmation_name, confirmation_count, confirmation_start_ms
            confirmation_name, confirmation_count, confirmation_start_ms = None, 0, None

        def confirm(name: str, condition: bool, timestamp_ms: int) -> tuple[int, int] | None:
            """Return (first candidate, confirmation frame) without backdating event time."""
            nonlocal confirmation_name, confirmation_count, confirmation_start_ms
            if not condition:
                clear_confirmation()
                return None
            if confirmation_name != name:
                confirmation_name, confirmation_count = name, 1
                confirmation_start_ms = timestamp_ms
            else:
                confirmation_count += 1
            if confirmation_count >= cfg.confirmation_frames:
                candidate_ms = int(confirmation_start_ms if confirmation_start_ms is not None else timestamp_ms)
                clear_confirmation()
                return candidate_ms, timestamp_ms
            return None

        def reset_attempt() -> None:
            nonlocal state, attempt_start_ms, takeoff_candidate_ms, takeoff_confirmation_ms
            nonlocal landing_candidate_ms, attempt_quality
            state = "STANDING"
            attempt_start_ms = takeoff_candidate_ms = takeoff_confirmation_ms = None
            landing_candidate_ms = None
            attempt_quality = []
            clear_confirmation()

        def reject_attempt(reason: str, end_timestamp_ms: int) -> dict[str, Any]:
            event = {
                "event_type": "rejected",
                "attempt_start_timestamp_ms": attempt_start_ms,
                "start_timestamp_ms": attempt_start_ms,
                "takeoff_candidate_timestamp_ms": takeoff_candidate_ms,
                "takeoff_confirmation_timestamp_ms": takeoff_confirmation_ms,
                "takeoff_timestamp_ms": takeoff_confirmation_ms,
                "landing_candidate_timestamp_ms": landing_candidate_ms,
                "landing_confirmation_timestamp_ms": None,
                "landing_timestamp_ms": None,
                "end_timestamp_ms": end_timestamp_ms,
                "reason": reason,
            }
            events.append(event)
            reset_attempt()
            return event

        for item in measurements:
            timestamp = int(item.timestamp_ms)
            event_start_index = len(events)
            foot = item.values.get("foot_ground_y")
            hip = item.values.get("hip_center_y")
            valid = bool(item.quality.get("measurement_valid")) and foot is not None and hip is not None

            if valid:
                foot, hip = float(foot), float(hip)
                score = item.quality.get("visibility_score")
                if score is not None:
                    quality_scores.append(float(score))
                    if state in active_states:
                        attempt_quality.append(float(score))
            else:
                clear_confirmation()
                if state == "SEEK_STANDING_BASELINE":
                    baseline.clear()
                if (last_valid_ms is not None
                        and (timestamp - last_valid_ms) / 1000.0 > cfg.max_measurement_gap_s
                        and state in active_states):
                    reject_attempt("long_measurement_gap", timestamp)
                rows.append({
                    "frame_number": item.frame_number, "timestamp_ms": timestamp,
                    "foot_ground_y": None, "hip_center_y": None,
                    "vertical_displacement": None, "hip_rise": None,
                    "attempt_start_timestamp_ms": attempt_start_ms,
                    "takeoff_candidate_timestamp_ms": takeoff_candidate_ms,
                    "takeoff_confirmation_timestamp_ms": takeoff_confirmation_ms,
                    "landing_candidate_timestamp_ms": landing_candidate_ms,
                    "state": state, "measurement_valid": False,
                    "events": events[event_start_index:],
                })
                continue

            if (last_valid_ms is not None
                    and (timestamp - last_valid_ms) / 1000.0 > cfg.max_measurement_gap_s
                    and state in active_states):
                reject_attempt("long_measurement_gap", timestamp)
            last_valid_ms = timestamp
            displacement = None if baseline_foot is None else baseline_foot - foot
            hip_rise = None if baseline_hip is None else baseline_hip - hip

            if state == "SEEK_STANDING_BASELINE":
                if startup_origin is None:
                    startup_origin = (foot, hip, timestamp)
                elif not startup_rejection_recorded and (
                    abs(startup_origin[0] - foot) >= cfg.takeoff_displacement
                    and abs(startup_origin[1] - hip) >= cfg.attempt_initialization_displacement
                ):
                    startup_event = {
                        "event_type": "rejected", "attempt_start_timestamp_ms": startup_origin[2],
                        "start_timestamp_ms": startup_origin[2], "takeoff_candidate_timestamp_ms": None,
                        "takeoff_confirmation_timestamp_ms": None, "takeoff_timestamp_ms": None,
                        "landing_candidate_timestamp_ms": None, "landing_confirmation_timestamp_ms": None,
                        "landing_timestamp_ms": None, "end_timestamp_ms": timestamp,
                        "reason": "video_started_without_standing_baseline",
                    }
                    events.append(startup_event)
                    startup_rejection_recorded = True

                if not baseline or (
                    max(abs(foot - x) for x, _ in baseline) <= cfg.baseline_max_variation
                    and max(abs(hip - y) for _, y in baseline) <= cfg.baseline_max_variation
                ):
                    baseline.append((foot, hip))
                else:
                    baseline = [(foot, hip)]
                if len(baseline) >= cfg.baseline_confirmation_frames:
                    baseline_foot = median(x for x, _ in baseline)
                    baseline_hip = median(y for _, y in baseline)
                    state = "STANDING"

            elif state == "STANDING":
                # Attempt start is a preparatory posture departure, not takeoff.
                if (abs(displacement or 0.0) >= cfg.attempt_initialization_displacement
                        or abs(hip_rise or 0.0) >= cfg.attempt_initialization_displacement):
                    attempt_start_ms = timestamp
                    attempt_quality = [float(score)] if score is not None else []
                    state = "ATTEMPT_INITIALIZED"

            elif state in {"ATTEMPT_INITIALIZED", "TAKEOFF_CANDIDATE"}:
                takeoff_condition = (displacement or 0.0) >= cfg.takeoff_displacement and (hip_rise or 0.0) > 0
                if not takeoff_condition:
                    clear_confirmation()
                    if state == "TAKEOFF_CANDIDATE":
                        state = "ATTEMPT_INITIALIZED"
                else:
                    confirmed = confirm("takeoff", True, timestamp)
                    if confirmed is None:
                        if confirmation_start_ms is not None:
                            takeoff_candidate_ms = confirmation_start_ms
                        state = "TAKEOFF_CANDIDATE"
                    else:
                        takeoff_candidate_ms, takeoff_confirmation_ms = confirmed
                        state = "FLIGHT"

            elif state in {"FLIGHT", "LANDING_CANDIDATE"}:
                landing_condition = (displacement or 0.0) <= cfg.landing_displacement
                if not landing_condition:
                    clear_confirmation()
                    if state == "LANDING_CANDIDATE":
                        state = "FLIGHT"
                else:
                    confirmed = confirm("landing", True, timestamp)
                    if confirmed is None:
                        if confirmation_start_ms is not None:
                            landing_candidate_ms = confirmation_start_ms
                        state = "LANDING_CANDIDATE"
                    else:
                        landing_candidate_ms, landing_confirmation_ms = confirmed
                        flight_s = ((landing_confirmation_ms - int(
                            takeoff_confirmation_ms if takeoff_confirmation_ms is not None else landing_confirmation_ms
                        )) / 1000.0)
                        attempt_s = ((landing_confirmation_ms - int(
                            attempt_start_ms if attempt_start_ms is not None else landing_confirmation_ms
                        )) / 1000.0)
                        reason = None
                        if flight_s < cfg.min_flight_duration_s:
                            reason = "flight_too_short"
                        elif flight_s > cfg.max_flight_duration_s:
                            reason = "flight_too_long"
                        elif attempt_s < cfg.min_attempt_duration_s:
                            reason = "attempt_too_short"
                        elif attempt_s > cfg.max_attempt_duration_s:
                            reason = "attempt_too_long"

                        if reason is not None:
                            reject_attempt(reason, landing_confirmation_ms)
                        else:
                            height_m = cfg.gravity_m_s2 * flight_s * flight_s / 8.0
                            accepted = {
                                "event_type": "accepted",
                                "attempt_number": 1 + sum(e["event_type"] == "accepted" for e in events),
                                "attempt_start_timestamp_ms": attempt_start_ms,
                                "start_timestamp_ms": attempt_start_ms,
                                "takeoff_candidate_timestamp_ms": takeoff_candidate_ms,
                                "takeoff_confirmation_timestamp_ms": takeoff_confirmation_ms,
                                "takeoff_timestamp_ms": takeoff_confirmation_ms,
                                "landing_candidate_timestamp_ms": landing_candidate_ms,
                                "landing_confirmation_timestamp_ms": landing_confirmation_ms,
                                "landing_timestamp_ms": landing_confirmation_ms,
                                "flight_time_seconds": round(flight_s, 4),
                                "flight_time_height_m": round(height_m, 5),
                                "flight_time_height_cm": round(height_m * 100.0, 2),
                                "metric_definition": "takeoff_to_apex_flight_time_height; not standing-to-apex or pose-derived center-of-mass displacement",
                                "mean_landmark_visibility": round(sum(attempt_quality) / len(attempt_quality), 4) if attempt_quality else None,
                                "reason": None,
                            }
                            events.append(accepted)
                            reset_attempt()

            if attempt_start_ms is not None and state in active_states and (
                timestamp - attempt_start_ms
            ) / 1000.0 > cfg.max_attempt_duration_s:
                reject_attempt("attempt_too_long", timestamp)

            rows.append({
                "frame_number": item.frame_number, "timestamp_ms": timestamp,
                "foot_ground_y": foot, "hip_center_y": hip,
                "vertical_displacement": displacement, "hip_rise": hip_rise,
                "attempt_start_timestamp_ms": attempt_start_ms,
                "takeoff_candidate_timestamp_ms": takeoff_candidate_ms,
                "takeoff_confirmation_timestamp_ms": takeoff_confirmation_ms,
                "landing_candidate_timestamp_ms": landing_candidate_ms,
                "state": state, "measurement_valid": True,
                "events": events[event_start_index:],
            })

        if state in active_states:
            end = measurements[-1].timestamp_ms if measurements else 0
            reject_attempt("incomplete_attempt", int(end))
        elif state == "SEEK_STANDING_BASELINE" and measurements and not startup_rejection_recorded:
            events.append({
                "event_type": "rejected", "attempt_start_timestamp_ms": None,
                "start_timestamp_ms": None, "takeoff_candidate_timestamp_ms": None,
                "takeoff_confirmation_timestamp_ms": None, "takeoff_timestamp_ms": None,
                "landing_candidate_timestamp_ms": None, "landing_confirmation_timestamp_ms": None,
                "landing_timestamp_ms": None, "end_timestamp_ms": measurements[-1].timestamp_ms,
                "reason": "standing_baseline_not_detected",
            })

        accepted_events = [event for event in events if event["event_type"] == "accepted"]
        rejected_events = [event for event in events if event["event_type"] == "rejected"]
        return AssessmentResult("vertical_jump", {
            "attempt_count": len(accepted_events), "measurements": rows,
            "accepted_attempts": accepted_events, "rejected_attempts": rejected_events,
            "measurement_coverage_percentage": round(100 * sum(bool(m.quality.get("measurement_valid")) for m in measurements) / len(measurements), 2) if measurements else 0,
            "mean_visibility": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else None,
            "confidence_note": "Visibility and signal coverage are quality indicators, not a validated probability of correctness.",
            "height_method": "flight_time_height_g_t_squared_over_8",
            "height_output_field": "flight_time_height_cm",
            "height_definition": "takeoff-to-apex flight-time height; not standing-to-apex or pose-derived center-of-mass displacement",
            "height_assumption": "center-of-mass heights at takeoff and confirmed landing are equal",
            "contact_detection_note": "Takeoff and landing timestamps are confirmation-frame times for foot/ankle landmark thresholds; they are video proxies, not force/contact sensor measurements.",
        }, tuple(events), asdict(cfg))
