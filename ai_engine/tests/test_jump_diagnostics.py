"""Synthetic checks for diagnostic row preparation and calibration input handling."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from athletelens_ai.assessments.base import AssessmentMeasurement
from broad_jump_diagnostic import calibration_from_input, estimate_start_heel, rows_for_csv as broad_rows
from vertical_jump_diagnostic import event_frame_index, rows_for_csv as vertical_rows


class JumpDiagnosticTests(unittest.TestCase):
    def test_vertical_runner_keeps_missing_gaps_and_timestamps(self):
        samples = [
            {"frame_number": 0, "timestamp_ms": 0, "state": "STANDING", "events": []},
            {"frame_number": 1, "timestamp_ms": 34, "state": "FLIGHT", "vertical_displacement": None, "events": []},
        ]
        observations = [SimpleNamespace(pose_detected=True), SimpleNamespace(pose_detected=False)]
        measurements = [
            AssessmentMeasurement(0, 0, {}, {"measurement_valid": True, "visibility_score": .9}),
            AssessmentMeasurement(1, 34, {}, {"measurement_valid": False, "visibility_score": None, "missing_landmarks": "feet"}),
        ]
        rows = vertical_rows(SimpleNamespace(summary={"measurements": samples}), observations, measurements)
        self.assertEqual([r["timestamp_ms"] for r in rows], [0, 34])
        self.assertIsNone(rows[1].get("vertical_displacement"))
        self.assertFalse(rows[1]["pose_detected"])
        self.assertEqual(event_frame_index(rows, 33), 1)

    def test_vertical_runner_exports_candidate_confirmation_and_rejection_details(self):
        event = {"event_type":"accepted", "attempt_number":1,
                 "start_timestamp_ms":500, "takeoff_candidate_timestamp_ms":700,
                 "takeoff_confirmation_timestamp_ms":800, "takeoff_timestamp_ms":800,
                 "landing_candidate_timestamp_ms":1100, "landing_confirmation_timestamp_ms":1200,
                 "landing_timestamp_ms":1200}
        raw_rows = [{"frame_number":i,"timestamp_ms":t,"state":"FLIGHT",
                     "events":[event] if t == 1200 else []} for i,t in enumerate((700,800,1100,1200))]
        obs = [SimpleNamespace(pose_detected=True) for _ in raw_rows]
        ms = [AssessmentMeasurement(i,t,{}, {"measurement_valid":True,"visibility_score":.9})
              for i,t in enumerate((700,800,1100,1200))]
        rows = vertical_rows(SimpleNamespace(summary={"measurements":raw_rows}),obs,ms,[event])
        self.assertEqual(rows[-1]["repetition_count"],1)
        annotations = rows[-1]["event_annotations"]
        self.assertEqual({a["type"] for a in annotations},
                         {"takeoff_candidate","takeoff_confirmed","landing_candidate","landing_confirmed","accepted"})

    def test_broad_calibration_input_valid_and_invalid(self):
        calibration, status = calibration_from_input([20, 120, 220, 120], 2.0, 640, 360)
        self.assertIsNotNone(calibration)
        self.assertEqual(status["status"], "valid")
        invalid, invalid_status = calibration_from_input([700, 120, 900, 120], 2.0, 640, 360)
        self.assertIsNone(invalid)
        self.assertEqual(invalid_status["status"], "invalid")
        self.assertFalse(invalid_status["physical_distance_available"])

    def test_broad_runner_preserves_pixel_only_measurements_without_rescaling(self):
        summary = {"measurements": [{"timestamp_ms": 100, "heel_center_x_px": 256.0,
                                      "heel_ground_y": .8, "horizontal_displacement_px": 128.0,
                                      "physical_distance_m": None, "physical_distance_cm": None,
                                      "physical_distance_available": False, "movement_state": "FLIGHT"}]}
        observations = [SimpleNamespace(pose_detected=True)]
        measurements = [AssessmentMeasurement(4, 100, {}, {"measurement_valid": True,
                         "visibility_score": .8, "selected_foot_landmarks": "29,30"})]
        rows = broad_rows(SimpleNamespace(summary=summary), observations, measurements, None)
        self.assertEqual(rows[0]["timestamp_ms"], 100)
        self.assertEqual(rows[0]["heel_center_x_px"], 256.0)
        self.assertEqual(rows[0]["horizontal_displacement_px"], 128.0)
        self.assertIsNone(rows[0]["physical_distance_m"])
        self.assertFalse(rows[0]["physical_distance_available"])

    def test_start_heel_marker_uses_stable_start_samples(self):
        rows = [{"heel_center_x_px": 10.0, "heel_ground_y": .8, "movement_state": "SEEK_STARTING_POSITION"},
                {"heel_center_x_px": 12.0, "heel_ground_y": .82, "movement_state": "STANDING"}]
        self.assertEqual(estimate_start_heel(rows, 2), (11.0, .81))


if __name__ == "__main__":
    unittest.main()
