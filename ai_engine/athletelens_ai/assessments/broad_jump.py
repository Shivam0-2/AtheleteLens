"""Experimental standing broad-jump analysis with explicit image calibration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .base import AssessmentMeasurement, AssessmentProtocol, AssessmentResult
from .calibration import CalibrationError, GroundReferenceCalibration
from ..pipeline import PoseObservation
from ..video import VideoMetadata

# MediaPipe PoseLandmarker indices: bilateral ankles, heels, foot indices.
_HEEL_IDS = (29, 30)
_ANKLE_IDS = (27, 28)
_FOOT_INDEX_IDS = (31, 32)


@dataclass(frozen=True)
class BroadJumpConfig:
    """Detector settings; horizontal distance minimum is an image-size fraction.

    ``minimum_horizontal_displacement_fraction`` is multiplied by the larger
    frame dimension to obtain the pixel threshold used by the detector.
    """

    min_visibility: float = 0.5
    baseline_confirmation_frames: int = 5
    confirmation_frames: int = 2
    baseline_max_variation_px: float = 8.0
    baseline_max_vertical_variation: float = 0.015
    takeoff_vertical_displacement: float = 0.025
    landing_vertical_tolerance: float = 0.015
    minimum_horizontal_displacement_fraction: float = 0.03
    max_measurement_gap_s: float = 0.25
    min_attempt_duration_s: float = 0.2
    max_attempt_duration_s: float = 5.0
    pixels_per_meter: float | None = None
    reference_length_m: float | None = None
    reference_length_pixels: float | None = None
    ground_reference: GroundReferenceCalibration | None = None

    def validate(self) -> None:
        if not math.isfinite(self.min_visibility) or not 0 <= self.min_visibility <= 1:
            raise ValueError("min_visibility must be between 0 and 1")
        if (isinstance(self.baseline_confirmation_frames, bool) or self.baseline_confirmation_frames < 2
                or isinstance(self.confirmation_frames, bool) or self.confirmation_frames < 1):
            raise ValueError("Confirmation frame counts are too small")
        for name in ("baseline_max_variation_px", "baseline_max_vertical_variation", "takeoff_vertical_displacement", "landing_vertical_tolerance", "max_measurement_gap_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (not math.isfinite(self.minimum_horizontal_displacement_fraction)
                or not 0 < self.minimum_horizontal_displacement_fraction <= 1):
            raise ValueError("minimum_horizontal_displacement_fraction must be in (0, 1]")
        if self.landing_vertical_tolerance >= self.takeoff_vertical_displacement:
            raise ValueError("Landing tolerance must be below takeoff threshold")
        if not math.isfinite(self.min_attempt_duration_s) or not math.isfinite(self.max_attempt_duration_s) or not (0 <= self.min_attempt_duration_s < self.max_attempt_duration_s):
            raise ValueError("Attempt duration limits must satisfy 0 <= min < max")
        direct = self.pixels_per_meter is not None
        reference = self.reference_length_m is not None or self.reference_length_pixels is not None
        if sum((direct, reference, self.ground_reference is not None)) > 1:
            raise ValueError("Use only one calibration source")
        if direct and (not math.isfinite(self.pixels_per_meter) or self.pixels_per_meter <= 0):
            raise ValueError("pixels_per_meter must be positive and finite")
        if reference:
            if self.reference_length_m is None or self.reference_length_pixels is None:
                raise ValueError("Both reference length fields are required")
            if not math.isfinite(self.reference_length_m) or self.reference_length_m <= 0 or not math.isfinite(self.reference_length_pixels) or self.reference_length_pixels <= 0:
                raise ValueError("Reference lengths must be positive and finite")
        if self.ground_reference is not None:
            self.ground_reference.validate_values()

    @property
    def resolved_pixels_per_meter(self) -> float | None:
        if self.pixels_per_meter is not None:
            return self.pixels_per_meter
        if self.reference_length_m is not None and self.reference_length_pixels is not None:
            return self.reference_length_pixels / self.reference_length_m
        return None


def _visible(point: Any, minimum: float) -> bool:
    try:
        values = float(point.x), float(point.y), float(point.visibility)
        presence = getattr(point, "presence", None)
        return all(math.isfinite(v) for v in values) and values[2] >= minimum and (presence is None or float(presence) >= minimum)
    except (AttributeError, TypeError, ValueError):
        return False


class BroadJumpAssessment(AssessmentProtocol):
    """Track heel-center movement in a fixed side-on camera view.

    A scalar pixels-per-meter value is valid only for a calibrated ground plane
    with the subject and reference at comparable camera depth.
    """

    def __init__(self, config: BroadJumpConfig | None = None):
        self.config = config or BroadJumpConfig()
        self.config.validate()

    def measure_frame(self, observation: PoseObservation) -> AssessmentMeasurement:
        points = observation.pose_landmarks[0] if observation.pose_detected else []
        ids = _HEEL_IDS
        feet = [points[i] for i in ids if i < len(points) and points[i] is not None and _visible(points[i], self.config.min_visibility)]
        if len(feet) < 2:
            ids = _ANKLE_IDS
            feet = [points[i] for i in ids if i < len(points) and points[i] is not None and _visible(points[i], self.config.min_visibility)]
        if len(feet) < 2:
            ids = _FOOT_INDEX_IDS
            feet = [points[i] for i in ids if i < len(points) and points[i] is not None and _visible(points[i], self.config.min_visibility)]
        valid = len(feet) >= 2
        # MediaPipe image-landmark x is normalized to [0, 1]. Convert it once
        # in analyze(), where the actual video width is available.
        heel_x_normalized = sum(float(p.x) for p in feet) / len(feet) if valid else None
        heel_y = max(float(p.y) for p in feet) if valid else None
        score = min(float(p.visibility) for p in feet) if valid else None
        return AssessmentMeasurement(observation.frame_number, observation.timestamp_ms,
            {"heel_center_x_normalized": heel_x_normalized, "heel_ground_y": heel_y,
             "selected_foot_landmarks": ",".join(str(i) for i in ids) if valid else ""},
            {"pose_detected": observation.pose_detected, "measurement_valid": valid,
             "visibility_score": score, "missing_landmarks": "bilateral_ankle_heel_foot" if not valid else "",
             "selected_foot_landmarks": ",".join(str(i) for i in ids) if valid else ""})

    def analyze(self, measurements: Sequence[AssessmentMeasurement], video: VideoMetadata) -> AssessmentResult:
        cfg = self.config
        state = "SEEK_STARTING_POSITION"
        baseline: list[tuple[float, float]] = []
        start_x = start_y = None
        last_valid = None
        confirm_name = None
        confirm_count = 0
        attempt_start = takeoff_ms = None
        max_displacement = 0.0
        events: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        quality_scores: list[float] = []
        attempt_quality: list[float] = []
        scale = cfg.resolved_pixels_per_meter
        calibration_result = None
        calibration_error = None
        if cfg.ground_reference is not None:
            try:
                calibration_result = cfg.ground_reference.calibrate(video.width, video.height)
                scale = calibration_result.pixels_per_meter
            except CalibrationError as exc:
                scale = None
                calibration_error = str(exc)
        if scale is not None:
            calibration_status = "valid"
            calibration_explanation = (
                "A scale is available; physical distance uses the local calibrated "
                "pixels-per-meter value. Perspective may change scale elsewhere in the frame."
            )
        elif calibration_error:
            calibration_status = "invalid"
            calibration_explanation = calibration_error
        else:
            calibration_status = "unavailable"
            calibration_explanation = "No valid calibration was supplied; the detected attempt is reported in pixels only."
        calibration_source = (
            "image_ground_reference" if calibration_result is not None else
            "direct_scale" if cfg.pixels_per_meter is not None else
            "known_reference" if scale is not None else
            "image_ground_reference" if cfg.ground_reference is not None else None
        )

        def confirmed(name: str, condition: bool) -> bool:
            nonlocal confirm_name, confirm_count
            if not condition:
                confirm_name, confirm_count = None, 0
                return False
            if name == confirm_name:
                confirm_count += 1
            else:
                confirm_name, confirm_count = name, 1
            if confirm_count >= cfg.confirmation_frames:
                confirm_name, confirm_count = None, 0
                return True
            return False

        def reject(reason: str, end: int, displacement: float | None = None) -> None:
            nonlocal state, attempt_start, takeoff_ms, max_displacement, attempt_quality
            events.append({"event_type": "rejected", "start_timestamp_ms": attempt_start,
                           "attempt_status": "rejected",
                           "takeoff_timestamp_ms": takeoff_ms, "end_timestamp_ms": end,
                           "horizontal_displacement_px": round(displacement, 2) if displacement is not None else None,
                           "horizontal_displacement_units": "px",
                           "physical_distance_available": False,
                           "distance_m": None, "distance_cm": None,
                           "calibration_status": calibration_status,
                           "calibration_explanation": calibration_explanation,
                           "reason": reason})
            state, attempt_start, takeoff_ms, max_displacement = "STANDING", None, None, 0.0
            attempt_quality = []

        for item in measurements:
            ts = item.timestamp_ms
            x_normalized, y = item.values.get("heel_center_x_normalized"), item.values.get("heel_ground_y")
            # Horizontal positions and displacements in temporal analysis are
            # pixels; vertical coordinates and their thresholds remain normalized.
            x = float(x_normalized) * video.width if x_normalized is not None else None
            valid = bool(item.quality.get("measurement_valid")) and x is not None and y is not None
            emitted = []
            if not valid:
                confirm_name, confirm_count = None, 0
                if state == "SEEK_STARTING_POSITION":
                    baseline.clear()
                if last_valid is not None and (ts - last_valid) / 1000 > cfg.max_measurement_gap_s and state in {"TAKEOFF_CANDIDATE", "FLIGHT", "LANDING_CANDIDATE"}:
                    reject("long_measurement_gap", ts, max_displacement)
                rows.append({"frame_number": item.frame_number, "timestamp_ms": ts, "heel_center_x_px": None,
                             "heel_ground_y": None, "horizontal_displacement_px": None, "movement_state": state,
                             "measurement_valid": False, "events": []})
                continue
            x, y = float(x), float(y)
            score = item.quality.get("visibility_score")
            if score is not None:
                quality_scores.append(float(score))
                if state in {"TAKEOFF_CANDIDATE", "FLIGHT", "LANDING_CANDIDATE"} or (state == "STANDING" and attempt_start is not None):
                    attempt_quality.append(float(score))
            if last_valid is not None and (ts - last_valid) / 1000 > cfg.max_measurement_gap_s and state in {"TAKEOFF_CANDIDATE", "FLIGHT", "LANDING_CANDIDATE"}:
                reject("long_measurement_gap", ts, max_displacement)
            last_valid = ts
            displacement = abs(x - start_x) if start_x is not None else None
            vertical_rise = start_y - y if start_y is not None else None
            if state == "SEEK_STARTING_POSITION":
                if not baseline or (
                    max(abs(x - bx) for bx, _ in baseline) <= cfg.baseline_max_variation_px
                    and max(abs(y - by) for _, by in baseline) <= cfg.baseline_max_vertical_variation
                ):
                    baseline.append((x, y))
                else:
                    baseline = [(x, y)]
                if len(baseline) >= cfg.baseline_confirmation_frames:
                    start_x = sum(p[0] for p in baseline) / len(baseline)
                    start_y = sum(p[1] for p in baseline) / len(baseline)
                    state = "STANDING"
            elif state == "STANDING":
                if (vertical_rise or 0) >= cfg.takeoff_vertical_displacement:
                    attempt_start = ts
                    attempt_quality = [float(score)] if score is not None else []
                    state = "TAKEOFF_CANDIDATE"
                    confirm_name, confirm_count = "takeoff", 1
            elif state == "TAKEOFF_CANDIDATE":
                if confirmed("takeoff", (vertical_rise or 0) >= cfg.takeoff_vertical_displacement):
                    takeoff_ms = ts
                    state = "FLIGHT"
            elif state == "FLIGHT":
                max_displacement = max(max_displacement, displacement or 0.0)
                if confirmed("landing", (vertical_rise or 0) <= cfg.landing_vertical_tolerance):
                    state = "LANDING_CANDIDATE"
                    confirm_name, confirm_count = "landed_stable", 1
            elif state == "LANDING_CANDIDATE":
                max_displacement = max(max_displacement, displacement or 0.0)
                if (vertical_rise or 0) > cfg.landing_vertical_tolerance:
                    state, confirm_name, confirm_count = "FLIGHT", None, 0
                elif confirmed("landed_stable", True):
                    duration = (ts - int(attempt_start or ts)) / 1000.0
                    landing_displacement = abs(x - float(start_x))
                    if duration < cfg.min_attempt_duration_s:
                        reason = "attempt_too_short"
                    elif duration > cfg.max_attempt_duration_s:
                        reason = "attempt_too_long"
                    elif landing_displacement < cfg.minimum_horizontal_displacement_fraction * max(video.width, video.height):
                        reason = "insufficient_horizontal_displacement"
                    else:
                        reason = None
                    if reason:
                        reject(reason, ts, landing_displacement)
                    else:
                        distance_m = None
                        if scale is not None:
                            distance_m = (calibration_result.distance_m(landing_displacement)
                                          if calibration_result is not None else landing_displacement / scale)
                        event = {"event_type": "accepted", "attempt_number": 1 + sum(e["event_type"] == "accepted" for e in events),
                                 "attempt_status": "accepted",
                                 "start_timestamp_ms": attempt_start, "takeoff_timestamp_ms": takeoff_ms,
                                 "landing_timestamp_ms": ts, "duration_seconds": round(duration, 4),
                                 "horizontal_displacement_px": round(landing_displacement, 2),
                                 "horizontal_displacement_units": "px",
                                 "physical_distance_available": distance_m is not None,
                                 "distance_m": round(distance_m, 4) if distance_m is not None else None,
                                 "distance_cm": round(distance_m * 100, 2) if distance_m is not None else None,
                                 "calibration_status": calibration_status,
                                 "calibration_explanation": calibration_explanation,
                                 "pixels_per_meter": round(scale, 6) if scale is not None else None,
                                 "mean_landmark_visibility": round(sum(attempt_quality) / len(attempt_quality), 4) if attempt_quality else None,
                                 "reason": None}
                        events.append(event); emitted.append(event)
                        state, attempt_start, takeoff_ms, max_displacement = "STANDING", None, None, 0.0
                        attempt_quality = []
            rows.append({"frame_number": item.frame_number, "timestamp_ms": ts, "heel_center_x_px": x,
                         "heel_ground_y": y, "horizontal_displacement_px": displacement,
                         "horizontal_displacement_units": "px",
                         "physical_distance_m": round(displacement / scale, 4) if displacement is not None and scale is not None else None,
                         "physical_distance_cm": round(displacement / scale * 100, 2) if displacement is not None and scale is not None else None,
                         "physical_distance_available": scale is not None,
                         "movement_state": state, "measurement_valid": True, "events": emitted})

        if state in {"TAKEOFF_CANDIDATE", "FLIGHT", "LANDING_CANDIDATE"}:
            reject("incomplete_attempt", measurements[-1].timestamp_ms if measurements else 0, max_displacement)
        elif state == "SEEK_STARTING_POSITION" and measurements:
            events.append({"event_type": "rejected", "start_timestamp_ms": None,
                           "attempt_status": "rejected",
                           "takeoff_timestamp_ms": None, "end_timestamp_ms": measurements[-1].timestamp_ms,
                           "horizontal_displacement_px": None, "distance_m": None, "distance_cm": None,
                           "horizontal_displacement_units": "px", "physical_distance_available": False,
                           "calibration_status": calibration_status,
                           "calibration_explanation": calibration_explanation,
                           "reason": "starting_position_not_detected"})
        accepted = [event for event in events if event["event_type"] == "accepted"]
        rejected = [event for event in events if event["event_type"] == "rejected"]
        return AssessmentResult("standing_broad_jump", {
            "attempt_count": len(accepted), "accepted_attempt_count": len(accepted),
            "physical_distance_attempt_count": sum(bool(event["physical_distance_available"]) for event in accepted),
            "attempts_with_physical_distance": [event for event in accepted if event["physical_distance_available"]],
            "measurements": rows, "accepted_attempts": accepted,
            "rejected_attempts": rejected,
            "rejection_reasons": [event["reason"] for event in rejected],
            "measurement_coverage_percentage": round(100 * sum(bool(m.quality.get("measurement_valid")) for m in measurements) / len(measurements), 2) if measurements else 0,
            "mean_visibility": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else None,
            "calibration": {"status": calibration_status, "valid": scale is not None,
                            "physical_distance_available": scale is not None,
                            "explanation": calibration_explanation,
                            "pixels_per_meter": scale,
                            "source": calibration_source,
                            "reference_length_m": cfg.reference_length_m, "reference_length_pixels": cfg.reference_length_pixels,
                            "ground_reference": calibration_result.to_dict() if calibration_result is not None else None,
                            "error": calibration_error,
                            "assumptions": "Fixed side-on camera; a flat reference near the landing path; local ground-plane scale only. Perspective can change scale elsewhere in the frame."},
            "measurement_units": {"horizontal_displacement_px": "px", "distance_m": "m", "distance_cm": "cm"},
            "physical_distance_available": scale is not None,
            "unscaled_displacements_px": [e["horizontal_displacement_px"] for e in events if e["horizontal_displacement_px"] is not None],
        }, tuple(events), asdict(cfg))
