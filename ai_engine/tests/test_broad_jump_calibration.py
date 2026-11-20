"""Ground reference calibration and safe distance conversion tests."""
from __future__ import annotations

import math
import unittest

from athletelens_ai.assessments.broad_jump import BroadJumpAssessment, BroadJumpConfig
from athletelens_ai.assessments.base import AssessmentMeasurement
from athletelens_ai.assessments.calibration import CalibrationError, GroundReferenceCalibration
from athletelens_ai.video import VideoMetadata


def synthetic_attempt():
    positions = [(100,.90)]*5 + [(105,.86),(125,.82),(150,.80),(180,.84),(200,.89),(200,.90),(200,.90)]
    return [AssessmentMeasurement(i,i*100,{"heel_center_x_normalized":x/640,"heel_ground_y":y},
        {"measurement_valid":True,"pose_detected":True,"visibility_score":.9})
        for i,(x,y) in enumerate(positions)]


class GroundReferenceCalibrationTests(unittest.TestCase):
    def test_reference_endpoints_compute_pixels_per_meter(self):
        calibration = GroundReferenceCalibration((100, 200), (300, 200), 2.0)
        result = calibration.calibrate(640, 360)
        self.assertEqual(result.pixels_per_meter, 100.0)
        self.assertEqual(result.reference_pixel_length, 200.0)
        self.assertEqual(result.scope, "local_ground_plane_near_reference")
        self.assertIn("perspective can change scale", result.perspective_note)

    def test_conversion_uses_valid_local_scale(self):
        result = GroundReferenceCalibration((10, 100), (210, 100), 2.0).calibrate(640, 360)
        self.assertAlmostEqual(result.distance_m(75), 0.75)
        with self.assertRaises(CalibrationError):
            result.distance_m(-1)
        with self.assertRaises(CalibrationError):
            result.distance_m(math.inf)

    def test_invalid_endpoint_and_scale_cases_are_rejected(self):
        invalid_cases = (
            GroundReferenceCalibration((-1, 20), (200, 20), 1.0),
            GroundReferenceCalibration((10, 20), (10, 20), 1.0),
            GroundReferenceCalibration((10, 20), (20, 200), 1.0),  # direction misaligned
            GroundReferenceCalibration((10, 20), (11, 20), 1.0),  # implausibly low scale
            GroundReferenceCalibration((10, 20), (210, 20), 1.0, max_pixels_per_meter=100.0),  # implausibly high scale
            GroundReferenceCalibration((10, 20), (200, 20), 0.0),
            GroundReferenceCalibration((math.nan, 20), (200, 20), 1.0),
        )
        for calibration in invalid_cases:
            with self.subTest(calibration=calibration), self.assertRaises(CalibrationError):
                calibration.calibrate(640, 360)
        with self.assertRaises(CalibrationError):
            GroundReferenceCalibration((10, 20), (210, 20), 1).calibrate(0, 360)

    def test_invalid_reference_keeps_completed_attempt_and_marks_distance_unavailable(self):
        # Endpoints are syntactically valid but outside this video's image bounds.
        reference = GroundReferenceCalibration((700, 100), (900, 100), 1.0)
        config = BroadJumpConfig(ground_reference=reference)
        result = BroadJumpAssessment(config).analyze(synthetic_attempt(), VideoMetadata(10, 640, 480))
        self.assertEqual(result.summary["attempt_count"], 1)
        self.assertFalse(result.summary["calibration"]["valid"])
        self.assertEqual(result.summary["calibration"]["status"], "invalid")
        self.assertEqual(result.summary["calibration"]["error"], "endpoint_a_px lies outside the video frame")
        accepted = result.summary["accepted_attempts"][0]
        self.assertEqual(accepted["attempt_status"], "accepted")
        self.assertEqual(accepted["calibration_status"], "invalid")
        self.assertIsNotNone(accepted["horizontal_displacement_px"])
        self.assertIsNone(accepted["distance_m"])
        self.assertIsNone(accepted["distance_cm"])
        self.assertFalse(accepted["physical_distance_available"])
        self.assertEqual(result.summary["rejected_attempts"], [])
        self.assertFalse(result.summary["physical_distance_available"])

    def test_valid_image_reference_drives_broad_jump_distance(self):
        reference = GroundReferenceCalibration((100, 100), (300, 100), 2.0)
        result = BroadJumpAssessment(BroadJumpConfig(ground_reference=reference)).analyze(
            synthetic_attempt(), VideoMetadata(10, 640, 480)
        )
        accepted = result.summary["accepted_attempts"][0]
        self.assertEqual(accepted["horizontal_displacement_px"], 100.0)
        self.assertEqual(accepted["distance_m"], 1.0)
        self.assertEqual(accepted["distance_cm"], 100.0)
        self.assertTrue(accepted["physical_distance_available"])
        self.assertEqual(accepted["horizontal_displacement_units"], "px")
        self.assertEqual(accepted["pixels_per_meter"], 100.0)
        self.assertEqual(result.summary["measurements"][-1]["physical_distance_m"], 1.0)
        self.assertEqual(result.summary["measurements"][-1]["physical_distance_cm"], 100.0)
        self.assertEqual(result.summary["calibration"]["source"], "image_ground_reference")
        self.assertTrue(result.summary["physical_distance_available"])

    def test_config_rejects_multiple_calibration_sources(self):
        with self.assertRaises(ValueError):
            BroadJumpConfig(pixels_per_meter=100, ground_reference=GroundReferenceCalibration((10, 10), (110, 10), 1)).validate()


if __name__ == "__main__":
    unittest.main()
