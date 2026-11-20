"""Synthetic regression tests for vertical-jump event timing and metric labels."""
from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from athletelens_ai.assessments.base import AssessmentMeasurement
from athletelens_ai.assessments.vertical_jump import VerticalJumpAssessment, VerticalJumpConfig
from athletelens_ai.pipeline import PoseObservation
from athletelens_ai.video import VideoMetadata


def m(frame: int, timestamp: int, foot: float | None, hip: float | None,
      valid: bool = True, visibility: float = .9) -> AssessmentMeasurement:
    return AssessmentMeasurement(
        frame, timestamp,
        {"foot_ground_y": foot, "hip_center_y": hip},
        {"measurement_valid": valid, "pose_detected": valid,
         "visibility_score": visibility if valid else None,
         "missing_landmarks": "" if valid else "feet_or_hips"},
    )


def standing(start_frame: int = 0, start_ms: int = 0) -> list[AssessmentMeasurement]:
    return [m(start_frame+i, start_ms+i*100, .90, .50) for i in range(5)]


def complete_jump(*, landing_hip: float = .50, delayed_takeoff: bool = False,
                  missing_takeoff: bool = False, missing_landing: bool = False) -> list[AssessmentMeasurement]:
    values = standing()
    values.extend([m(5,500,.90,.52), m(6,600,.90,.51)])  # attempt initialized during preparation
    if delayed_takeoff:
        values.extend([m(7,700,.90,.52), m(8,800,.87,.46), m(9,900,.85,.43)])
        takeoff_index = 10
    else:
        values.extend([m(7,700,.87,.46)])
        if missing_takeoff:
            values.append(m(8,750,None,None,False))
            values.extend([m(9,800,.85,.43), m(10,900,.83,.41)])
            takeoff_index = 11
        else:
            values.append(m(8,800,.85,.43))
            takeoff_index = 9
    # Airborne, then feet return to the baseline. Landing is confirmed on its
    # second valid contact-proxy frame, not backdated to the candidate frame.
    base_frame = values[-1].frame_number + 1
    base_time = values[-1].timestamp_ms + 100
    values.extend([m(base_frame,base_time,.83,.41),
                   m(base_frame+1,base_time+100,.86,.45),
                   m(base_frame+2,base_time+200,.89,landing_hip)])
    if missing_landing:
        values.append(m(base_frame+3,base_time+250,None,None,False))
        values.extend([m(base_frame+4,base_time+300,.90,landing_hip),
                       m(base_frame+5,base_time+400,.90,landing_hip)])
    else:
        values.append(m(base_frame+3,base_time+300,.90,landing_hip))
    return values


def analyze(values: list[AssessmentMeasurement]):
    return VerticalJumpAssessment().analyze(values, VideoMetadata(10,640,480))


class VerticalJumpEventTimingTests(unittest.TestCase):
    def test_normal_complete_jump_uses_confirmation_timestamps_and_flight_height_name(self):
        # Timestamp spacing intentionally differs from metadata FPS to prove
        # flight time comes from observation timestamps, not frame count/FPS.
        result = VerticalJumpAssessment().analyze(complete_jump(), VideoMetadata(30,640,480))
        self.assertEqual(result.summary["attempt_count"], 1)
        event = result.summary["accepted_attempts"][0]
        self.assertEqual(event["attempt_start_timestamp_ms"], 500)
        self.assertEqual(event["takeoff_candidate_timestamp_ms"], 700)
        self.assertEqual(event["takeoff_timestamp_ms"], 800)
        self.assertEqual(event["takeoff_confirmation_timestamp_ms"], 800)
        self.assertEqual(event["landing_candidate_timestamp_ms"], 1100)
        self.assertEqual(event["landing_timestamp_ms"], 1200)
        self.assertEqual(event["landing_confirmation_timestamp_ms"], 1200)
        self.assertEqual(event["flight_time_seconds"], .4)
        self.assertAlmostEqual(event["flight_time_height_cm"], 9.80665*.4*.4/8*100, places=2)
        self.assertNotIn("jump_height_cm", event)
        self.assertEqual(result.summary["height_output_field"], "flight_time_height_cm")
        serialized = json.loads(json.dumps(asdict(result)))
        self.assertEqual(serialized["events"][0]["takeoff_confirmation_timestamp_ms"], 800)

    def test_attempt_start_is_separate_from_delayed_takeoff(self):
        event = analyze(complete_jump(delayed_takeoff=True)).summary["accepted_attempts"][0]
        self.assertEqual(event["attempt_start_timestamp_ms"], 500)
        self.assertEqual(event["takeoff_candidate_timestamp_ms"], 800)
        self.assertEqual(event["takeoff_timestamp_ms"], 900)
        self.assertGreater(event["takeoff_timestamp_ms"], event["attempt_start_timestamp_ms"])

    def test_video_beginning_airborne_is_rejected_without_counting(self):
        samples = [(.84,.44),(.81,.41),(.78,.38),(.75,.35),(.79,.39),(.85,.46)]
        values = [m(i,i*100,foot,hip) for i,(foot,hip) in enumerate(samples)]
        values.extend(m(i, i*100, .90,.50) for i in range(6,15))
        result = analyze(values)
        self.assertEqual(result.summary["attempt_count"], 0)
        self.assertIn("video_started_without_standing_baseline", [e["reason"] for e in result.summary["rejected_attempts"]])

    def test_different_landing_posture_does_not_change_event_time_formula(self):
        result = analyze(complete_jump(landing_hip=.57))
        event = result.summary["accepted_attempts"][0]
        self.assertEqual(event["landing_timestamp_ms"], 1200)
        self.assertAlmostEqual(event["flight_time_height_cm"], 19.61, places=2)
        self.assertIn("equal", result.summary["height_assumption"])

    def test_missing_frame_around_takeoff_restarts_confirmation(self):
        result = analyze(complete_jump(missing_takeoff=True))
        event = result.summary["accepted_attempts"][0]
        self.assertEqual(event["takeoff_candidate_timestamp_ms"], 800)
        self.assertEqual(event["takeoff_timestamp_ms"], 900)
        self.assertFalse(result.summary["measurements"][8]["measurement_valid"])

    def test_missing_frame_around_landing_restarts_confirmation(self):
        result = analyze(complete_jump(missing_landing=True))
        event = result.summary["accepted_attempts"][0]
        self.assertEqual(event["landing_candidate_timestamp_ms"], 1200)
        self.assertEqual(event["landing_timestamp_ms"], 1300)
        self.assertFalse(any(e["reason"] == "long_measurement_gap" for e in result.events))

    def test_incomplete_jump_and_long_measurement_gap_are_rejected(self):
        incomplete = analyze(complete_jump()[:10])
        self.assertEqual(incomplete.summary["attempt_count"], 0)
        self.assertIn("incomplete_attempt", [e["reason"] for e in incomplete.events])

        values = standing()
        values.extend([m(5,500,.90,.52),m(6,600,.90,.51),m(7,700,.87,.46),m(8,800,.85,.43),
                       m(9,1200,None,None,False),m(10,1300,.83,.41)])
        gap_result = analyze(values)
        self.assertIn("long_measurement_gap", [e["reason"] for e in gap_result.events])
        self.assertEqual(gap_result.summary["attempt_count"], 0)

    def test_measurement_landmarks_and_low_visibility_remain_missing(self):
        landmarks = [None]*33
        for index in (23,24,27,28,29,30,31,32):
            landmarks[index] = SimpleNamespace(x=.5,y=.8,visibility=.1 if index >= 27 else .9)
        observation = PoseObservation(2,200,[landmarks],[],[],[],SimpleNamespace())
        measurement = VerticalJumpAssessment().measure_frame(observation)
        self.assertFalse(measurement.quality["measurement_valid"])
        self.assertIsNone(measurement.values["foot_ground_y"])

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            VerticalJumpConfig(attempt_initialization_displacement=0).validate()
        with self.assertRaises(ValueError):
            VerticalJumpConfig(confirmation_frames=0).validate()


if __name__ == "__main__":
    unittest.main()
