"""Ground-plane scale calibration from a visible known-length reference.

This supplies a local image scale only. It does not correct perspective or
make the scale valid elsewhere in the frame or at a different camera depth.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


class CalibrationError(ValueError):
    """Raised when image reference points cannot define a usable scale."""


@dataclass(frozen=True)
class GroundReferenceCalibration:
    """Image endpoints of a known physical reference lying flat on the ground.

    The reference should be parallel to the athlete's travel direction and
    close to the landing path in camera depth. Coordinates are pixel ``(x,y)``.
    """

    endpoint_a_px: tuple[float, float]
    endpoint_b_px: tuple[float, float]
    known_length_m: float
    min_pixels_per_meter: float = 10.0
    max_pixels_per_meter: float = 100_000.0
    max_axis_deviation_deg: float = 15.0
    min_reference_pixel_length_px: float = 2.0

    def validate_values(self) -> None:
        for name, point in (("endpoint_a_px", self.endpoint_a_px), ("endpoint_b_px", self.endpoint_b_px)):
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise CalibrationError(f"{name} must contain image x and y coordinates")
            try:
                finite_coordinates = all(math.isfinite(float(value)) for value in point)
            except (TypeError, ValueError):
                finite_coordinates = False
            if not finite_coordinates:
                raise CalibrationError(f"{name} must contain finite coordinates")
        try:
            valid_length = math.isfinite(float(self.known_length_m)) and float(self.known_length_m) > 0
        except (TypeError, ValueError):
            valid_length = False
        if not valid_length:
            raise CalibrationError("known_length_m must be positive and finite")
        try:
            min_scale = float(self.min_pixels_per_meter)
            max_scale = float(self.max_pixels_per_meter)
            valid_limits = (math.isfinite(min_scale) and math.isfinite(max_scale)
                            and min_scale > 0 and max_scale > min_scale)
        except (TypeError, ValueError):
            valid_limits = False
        if not valid_limits:
            raise CalibrationError("Pixel-per-meter limits are invalid")
        try:
            axis_limit = float(self.max_axis_deviation_deg)
            valid_axis_limit = math.isfinite(axis_limit) and 0 <= axis_limit <= 90
        except (TypeError, ValueError):
            valid_axis_limit = False
        if not valid_axis_limit:
            raise CalibrationError("max_axis_deviation_deg must be in [0, 90]")
        try:
            min_reference_length = float(self.min_reference_pixel_length_px)
            valid_reference_length = math.isfinite(min_reference_length) and min_reference_length > 0
        except (TypeError, ValueError):
            valid_reference_length = False
        if not valid_reference_length:
            raise CalibrationError("min_reference_pixel_length_px must be positive and finite")

        # Reject near-coincident clicks even if a tiny declared reference length
        # would otherwise make their calculated pixels-per-meter look plausible.
        ax, ay = (float(value) for value in self.endpoint_a_px)
        bx, by = (float(value) for value in self.endpoint_b_px)
        if math.hypot(bx - ax, by - ay) < float(self.min_reference_pixel_length_px):
            raise CalibrationError("Reference endpoints are too close for reliable calibration")

    def calibrate(self, image_width: int, image_height: int) -> "GroundCalibrationResult":
        """Validate frame coordinates and return the local pixels-per-meter scale."""
        self.validate_values()
        if image_width <= 0 or image_height <= 0:
            raise CalibrationError("Image dimensions must be positive")
        ax, ay = (float(value) for value in self.endpoint_a_px)
        bx, by = (float(value) for value in self.endpoint_b_px)
        for label, x, y in (("endpoint_a_px", ax, ay), ("endpoint_b_px", bx, by)):
            if not (0 <= x < image_width and 0 <= y < image_height):
                raise CalibrationError(f"{label} lies outside the video frame")
        dx, dy = bx - ax, by - ay
        pixel_length = math.hypot(dx, dy)
        deviation = math.degrees(math.atan2(abs(dy), abs(dx)))
        if deviation > float(self.max_axis_deviation_deg):
            raise CalibrationError("Reference must align with the image horizontal travel axis")
        scale = pixel_length / float(self.known_length_m)
        if not float(self.min_pixels_per_meter) <= scale <= float(self.max_pixels_per_meter):
            raise CalibrationError("Computed pixels-per-meter is outside configured plausible limits")
        return GroundCalibrationResult(
            pixels_per_meter=scale,
            reference_pixel_length=pixel_length,
            known_length_m=float(self.known_length_m),
            endpoint_a_px=(ax, ay),
            endpoint_b_px=(bx, by),
            image_width=image_width,
            image_height=image_height,
            scope="local_ground_plane_near_reference",
            perspective_note=(
                "Scale is local to the reference plane and its camera depth; "
                "perspective can change scale elsewhere in the frame."
            ),
        )


@dataclass(frozen=True)
class GroundCalibrationResult:
    pixels_per_meter: float
    reference_pixel_length: float
    known_length_m: float
    endpoint_a_px: tuple[float, float]
    endpoint_b_px: tuple[float, float]
    image_width: int
    image_height: int
    scope: str
    perspective_note: str

    def distance_m(self, displacement_px: float) -> float:
        """Convert displacement using this local scale after validating input."""
        if not math.isfinite(displacement_px) or displacement_px < 0:
            raise CalibrationError("Pixel displacement must be finite and non-negative")
        return displacement_px / self.pixels_per_meter

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
