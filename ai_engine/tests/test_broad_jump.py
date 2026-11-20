"""Synthetic tests for the experimental standing broad-jump assessment."""
from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from athletelens_ai.assessments.base import AssessmentMeasurement
from athletelens_ai.assessments.broad_jump import BroadJumpAssessment, BroadJumpConfig
from athletelens_ai.pipeline import PoseObservation
from athletelens_ai.video import VideoMetadata


def measurement(i,t,x,y,valid=True):
    # Synthetic coordinates are built in pixels, then encoded as MediaPipe's
    # normalized x value for a 640 px wide frame.
    normalized_x = None if x is None else x / 640
    return AssessmentMeasurement(i,t,{"heel_center_x_normalized":normalized_x,"heel_ground_y":y},
        {"measurement_valid":valid,"pose_detected":valid,"visibility_score":.9 if valid else None,
         "missing_landmarks":"" if valid else "feet"})


def attempt():
    # Stable stance; rise, horizontal travel, and settle at a new ground position.
    points=[(100,.90)]*5 + [(105,.86),(125,.82),(150,.80),(180,.84),(200,.89),(200,.90),(200,.90)]
    return [measurement(i,i*100,x,y) for i,(x,y) in enumerate(points)]


class BroadJumpTests(unittest.TestCase):
    def test_calibrated_attempt_reports_distance_units_and_timestamps(self):
        cfg=BroadJumpConfig(pixels_per_meter=100)
        result=BroadJumpAssessment(cfg).analyze(attempt(),VideoMetadata(10,640,480))
        self.assertEqual(result.summary["attempt_count"],1)
        event=result.summary["accepted_attempts"][0]
        self.assertEqual(event["landing_timestamp_ms"],1100)
        self.assertEqual(event["distance_cm"],100.0)
        self.assertEqual(event["attempt_status"],"accepted")
        self.assertEqual(event["horizontal_displacement_units"],"px")
        self.assertTrue(event["physical_distance_available"])
        self.assertNotIn("units",event)
        serialized=json.loads(json.dumps(asdict(result)))
        self.assertEqual(serialized["assessment_type"],"standing_broad_jump")
        self.assertEqual(serialized["summary"]["measurement_units"],
                         {"horizontal_displacement_px":"px","distance_m":"m","distance_cm":"cm"})

    def test_complete_attempt_without_calibration_is_accepted_in_pixels_only(self):
        result=BroadJumpAssessment().analyze(attempt(),VideoMetadata(10,640,480))
        self.assertEqual(result.summary["accepted_attempt_count"],1)
        self.assertEqual(result.summary["physical_distance_attempt_count"],0)
        accepted=result.summary["accepted_attempts"][0]
        self.assertEqual(accepted["attempt_status"],"accepted")
        self.assertAlmostEqual(accepted["horizontal_displacement_px"],100)
        self.assertEqual(accepted["horizontal_displacement_units"],"px")
        self.assertFalse(accepted["physical_distance_available"])
        self.assertIsNone(accepted["distance_m"])
        self.assertIsNone(accepted["distance_cm"])
        self.assertEqual(accepted["calibration_status"],"unavailable")
        self.assertIn("No valid calibration",accepted["calibration_explanation"])
        self.assertEqual(result.summary["rejected_attempts"],[])
        self.assertEqual(result.summary["calibration"]["status"],"unavailable")

    def test_known_reference_calibration_produces_cm(self):
        cfg=BroadJumpConfig(reference_length_m=1.0,reference_length_pixels=200.0)
        result=BroadJumpAssessment(cfg).analyze(attempt(),VideoMetadata(10,640,480))
        self.assertEqual(result.summary["accepted_attempts"][0]["distance_cm"],50.0)
        self.assertTrue(result.summary["accepted_attempts"][0]["physical_distance_available"])
        self.assertEqual(result.summary["calibration"]["source"],"known_reference")

    def test_incomplete_attempt_and_long_gap_are_rejected(self):
        incomplete=BroadJumpAssessment(BroadJumpConfig(pixels_per_meter=100)).analyze(attempt()[:9],VideoMetadata(10,640,480))
        self.assertIn("incomplete_attempt",[e["reason"] for e in incomplete.events])
        values=attempt(); values.insert(8,measurement(8,1200,None,None,False)); values[9]=measurement(9,1500,200,.90)
        result=BroadJumpAssessment(BroadJumpConfig(pixels_per_meter=100)).analyze(values,VideoMetadata(10,640,480))
        self.assertIn("long_measurement_gap",[e["reason"] for e in result.events])

    def test_missing_measurements_noise_and_invalid_calibration(self):
        invalid=[measurement(i,i*100,100+(i%2),.90) for i in range(20)]
        self.assertEqual(BroadJumpAssessment(BroadJumpConfig(pixels_per_meter=100)).analyze(invalid,VideoMetadata(10,640,480)).summary["attempt_count"],0)
        with self.assertRaises(ValueError): BroadJumpConfig(pixels_per_meter=0).validate()
        with self.assertRaises(ValueError): BroadJumpConfig(reference_length_m=1).validate()
        with self.assertRaises(ValueError): BroadJumpConfig(pixels_per_meter=100,reference_length_m=1,reference_length_pixels=100).validate()
        with self.assertRaises(ValueError): BroadJumpConfig(minimum_horizontal_displacement_fraction=1.1).validate()
        points=[None]*33
        for index in (29,30): points[index]=SimpleNamespace(x=.4,y=.8,visibility=.2)
        obs=PoseObservation(7,700,[points],[],[],[],SimpleNamespace())
        measured=BroadJumpAssessment().measure_frame(obs)
        self.assertEqual(measured.frame_number,7)
        self.assertEqual(measured.timestamp_ms,700)
        self.assertFalse(measured.quality["measurement_valid"])
        self.assertIsNone(measured.values["heel_center_x_normalized"])

    def test_normalized_coordinates_convert_to_pixels_using_video_width(self):
        values = [measurement(i, i * 100, x, y) for i, (x, y) in enumerate(
            [(100, .90)] * 5 + [(105, .86), (125, .82), (150, .80), (200, .84),
                               (200, .89), (200, .90), (200, .90)]
        )]
        result = BroadJumpAssessment(BroadJumpConfig(pixels_per_meter=100)).analyze(
            values, VideoMetadata(10, 1280, 720)
        )
        self.assertAlmostEqual(result.summary["accepted_attempts"][0]["horizontal_displacement_px"], 200)
        self.assertAlmostEqual(result.summary["measurements"][5]["heel_center_x_px"], 210)

    def test_pixel_displacement_threshold_uses_pixel_units(self):
        def make_attempt(displacement_px):
            xs_ys = [(100, .90)] * 5 + [(105, .86), (125, .82), (150, .80),
                                        (100 + displacement_px, .84),
                                        (100 + displacement_px, .89),
                                        (100 + displacement_px, .90),
                                        (100 + displacement_px, .90)]
            return [measurement(i, i * 100, x, y) for i, (x, y) in enumerate(xs_ys)]
        config = BroadJumpConfig(pixels_per_meter=100)
        accepted = BroadJumpAssessment(config).analyze(make_attempt(25), VideoMetadata(10, 640, 480))
        rejected = BroadJumpAssessment(config).analyze(make_attempt(18), VideoMetadata(10, 640, 480))
        self.assertEqual(accepted.summary["attempt_count"], 1)
        self.assertEqual(accepted.summary["accepted_attempts"][0]["horizontal_displacement_px"], 25)
        self.assertEqual(accepted.summary["accepted_attempts"][0]["distance_m"], .25)
        self.assertEqual(accepted.summary["accepted_attempts"][0]["distance_cm"], 25)
        self.assertEqual(rejected.summary["rejection_reasons"], ["insufficient_horizontal_displacement"])


if __name__ == "__main__": unittest.main()
