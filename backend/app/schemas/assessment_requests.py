"""Validated per-exercise options received as multipart JSON form fields."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from athletelens_ai.assessments.push_up import DetectorConfig


class _RequestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PushUpRequestConfig(_RequestConfig):
    bottom_angle_deg: float = Field(default=90, ge=0, le=180)
    top_angle_deg: float = Field(default=160, ge=0, le=180)
    hysteresis_deg: float = Field(default=5, ge=0)
    confirmation_frames: int = Field(default=2, ge=1)
    min_repetition_duration_s: float = Field(default=0.25, ge=0)
    max_repetition_duration_s: float = Field(default=3, gt=0)
    max_measurement_gap_s: float = Field(default=0.25, gt=0)

    @model_validator(mode="after")
    def validate_detector(self) -> PushUpRequestConfig:
        DetectorConfig(**self.model_dump()).validate()
        return self


class CurlUpRequestConfig(_RequestConfig):
    # The repository SRS defines no standardized curl-up thresholds, so these
    # two angles are required from the caller rather than silently defaulted.
    extended_angle_deg: float = Field(ge=0, le=180)
    flexed_angle_deg: float = Field(ge=0, le=180)
    hysteresis_deg: float = Field(default=5, ge=0)
    confirmation_frames: int = Field(default=2, ge=1)
    min_repetition_duration_s: float = Field(default=0.25, ge=0)
    max_repetition_duration_s: float = Field(default=3, gt=0)
    max_measurement_gap_s: float = Field(default=0.25, gt=0)
    min_visibility: float = Field(default=0.5, ge=0, le=1)
    starting_position: Literal["extended"] = "extended"
    selected_side: Literal["left", "right", "auto"] = "auto"
    test_duration_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_detector(self) -> CurlUpRequestConfig:
        from athletelens_ai.assessments.curl_up import CurlUpConfig

        CurlUpConfig(image_aspect_ratio=1.0, **self.model_dump()).validate()
        return self


class VerticalJumpRequestConfig(_RequestConfig):
    gravity_m_s2: float = Field(default=9.80665, gt=0)
    min_visibility: float = Field(default=0.5, ge=0, le=1)
    baseline_confirmation_frames: int = Field(default=5, ge=2)
    baseline_max_variation: float = Field(default=0.015, gt=0)
    attempt_initialization_displacement: float = Field(default=0.01, gt=0)
    takeoff_displacement: float = Field(default=0.025, gt=0)
    landing_displacement: float = Field(default=0.012, gt=0)
    confirmation_frames: int = Field(default=2, ge=1)
    max_measurement_gap_s: float = Field(default=0.25, gt=0)
    min_flight_duration_s: float = Field(default=0.12, gt=0)
    max_flight_duration_s: float = Field(default=1.5, gt=0)
    min_attempt_duration_s: float = Field(default=0.2, ge=0)
    max_attempt_duration_s: float = Field(default=5, gt=0)

    @model_validator(mode="after")
    def validate_detector(self) -> VerticalJumpRequestConfig:
        from athletelens_ai.assessments.vertical_jump import VerticalJumpConfig

        VerticalJumpConfig(**self.model_dump()).validate()
        return self


class BroadJumpRequestConfig(_RequestConfig):
    min_visibility: float = Field(default=0.5, ge=0, le=1)
    baseline_confirmation_frames: int = Field(default=5, ge=2)
    confirmation_frames: int = Field(default=2, ge=1)
    baseline_max_variation_px: float = Field(default=8, gt=0)
    baseline_max_vertical_variation: float = Field(default=0.015, gt=0)
    takeoff_vertical_displacement: float = Field(default=0.025, gt=0)
    landing_vertical_tolerance: float = Field(default=0.015, gt=0)
    minimum_horizontal_displacement_fraction: float = Field(default=0.03, gt=0, le=1)
    max_measurement_gap_s: float = Field(default=0.25, gt=0)
    min_attempt_duration_s: float = Field(default=0.2, ge=0)
    max_attempt_duration_s: float = Field(default=5, gt=0)
    reference_endpoint_a_x_px: float | None = None
    reference_endpoint_a_y_px: float | None = None
    reference_endpoint_b_x_px: float | None = None
    reference_endpoint_b_y_px: float | None = None
    reference_length_m: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_calibration_fields(self) -> BroadJumpRequestConfig:
        calibration_fields = (
            self.reference_endpoint_a_x_px,
            self.reference_endpoint_a_y_px,
            self.reference_endpoint_b_x_px,
            self.reference_endpoint_b_y_px,
            self.reference_length_m,
        )
        if any(value is not None for value in calibration_fields) and not all(
            value is not None for value in calibration_fields
        ):
            raise ValueError("All four reference endpoint coordinates and reference_length_m are required together")
        from athletelens_ai.assessments.broad_jump import BroadJumpConfig

        detector_values = self.model_dump(exclude={
            "reference_endpoint_a_x_px", "reference_endpoint_a_y_px",
            "reference_endpoint_b_x_px", "reference_endpoint_b_y_px", "reference_length_m",
        })
        BroadJumpConfig(**detector_values).validate()
        return self
