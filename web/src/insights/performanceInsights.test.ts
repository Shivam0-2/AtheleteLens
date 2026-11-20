import { describe, expect, it } from 'vitest'
import { generatePerformanceInsights } from './performanceInsights'
import type { AssessmentResponse, ProtocolAssessmentResponse, PushUpAssessmentResponse } from '../types'

function pushup(overrides: Partial<PushUpAssessmentResponse> = {}): AssessmentResponse {
  return {
    processing_status: 'completed', input_filename: 'sample.mp4', exercise_type: 'push_up',
    video: { fps: 30, width: 640, height: 360 }, total_frames_processed: 100,
    frames_with_detected_poses: 90, frames_with_valid_elbow_angle: 80,
    pose_detection_percentage: 90, measurement_coverage_percentage: 80,
    detected_repetitions: 2, average_elbow_angle_degrees: 110,
    minimum_elbow_angle_degrees: 75, maximum_elbow_angle_degrees: 160,
    accepted_repetition_timestamps: [{ duration_seconds: 1.2 }, { duration_seconds: 1.6 }],
    rejected_events: [], measurement_summary: {}, ...overrides,
  }
}

function protocol(exercise_type: ProtocolAssessmentResponse['exercise_type'], overrides: Partial<ProtocolAssessmentResponse> = {}): AssessmentResponse {
  return {
    processing_status: 'completed', input_filename: 'sample.mp4', exercise_type,
    video: { fps: 30, width: 640, height: 360 }, total_frames_processed: 100,
    frames_with_detected_poses: 100, frames_with_valid_measurements: 90,
    pose_detection_percentage: 100, measurement_coverage_percentage: 90,
    measurement_quality: {}, accepted_attempt_count: 1,
    accepted_attempts: [], rejected_attempt_count: 0, rejected_attempts: [],
    measurement_summary: {}, physical_distance_attempt_count: null,
    physical_distance_available: null, detector_configuration: {},
    annotated_video_available: false, annotated_video_path: null,
    ...overrides,
  }
}

describe('exercise-specific performance insights', () => {
  it('uses push-up repetition, angle range, and duration fields without grading', () => {
    const result = generatePerformanceInsights(pushup())
    expect(result.strengths.some((entry) => entry.evidenceFields.includes('detected_repetitions'))).toBe(true)
    expect(result.explanations.some((entry) => entry.evidenceFields.includes('minimum_elbow_angle_degrees'))).toBe(true)
    expect(result.explanations.some((entry) => entry.detail.includes('range 0.40 seconds'))).toBe(true)
    expect(JSON.stringify(result)).not.toMatch(/excellent|beginner|weak/i)
  })

  it('uses curl-up average and duration, and states that angle range is unavailable', () => {
    const response = protocol('curl_up', { measurement_summary: { average_included_angle_deg: 108 }, accepted_attempts: [{ duration_seconds: 1 }, { duration_seconds: 1.4 }] })
    const result = generatePerformanceInsights(response)
    expect(result.explanations.some((entry) => entry.evidenceFields.includes('measurement_summary.average_included_angle_deg'))).toBe(true)
    expect(result.limitations.some((entry) => entry.title.includes('Per-frame curl-up angle range'))).toBe(true)
    expect(result.explanations.some((entry) => entry.evidenceFields.includes('accepted_attempts'))).toBe(true)
  })

  it('uses vertical flight-time heights and identifies variation as descriptive', () => {
    const response = protocol('vertical_jump', { accepted_attempts: [{ flight_time_height_cm: 22 }, { flight_time_height_cm: 26 }] })
    const result = generatePerformanceInsights(response)
    expect(result.explanations.some((entry) => entry.detail.includes('22 to 26 cm'))).toBe(true)
    expect(result.limitations.some((entry) => entry.title === 'Height method and assumption')).toBe(true)
  })

  it('keeps uncalibrated broad-jump pixel displacement separate from unavailable distance', () => {
    const response = protocol('standing_broad_jump', { physical_distance_available: false, accepted_attempts: [{ attempt_status: 'accepted', horizontal_displacement_px: 210, physical_distance_available: false, distance_cm: null, distance_m: null }], measurement_summary: { calibration: { status: 'unavailable' } } })
    const result = generatePerformanceInsights(response)
    expect(result.explanations.some((entry) => entry.title === 'Physical distance unavailable')).toBe(true)
    expect(result.improvementAreas.some((entry) => entry.title.includes('Calibration is needed'))).toBe(true)
    expect(JSON.stringify(result)).not.toContain('210 cm')
  })

  it('handles missing measurements, zero attempts, rejected cycles, and poor tracking quality', () => {
    const response = protocol('vertical_jump', {
      accepted_attempt_count: 0, accepted_attempts: [],
      rejected_attempt_count: 1, rejected_attempts: [{ reason: 'long_measurement_gap' }],
      measurement_coverage_percentage: 0, pose_detection_percentage: 35,
    })
    const result = generatePerformanceInsights(response)
    expect(result.improvementAreas.some((entry) => entry.title.includes('No accepted cycles'))).toBe(true)
    expect(result.improvementAreas.some((entry) => entry.title.includes('rejected'))).toBe(true)
    expect(result.improvementAreas.some((entry) => entry.title.includes('No valid measurements'))).toBe(true)
    expect(result.improvementAreas.some((entry) => entry.title.includes('no detected pose'))).toBe(true)
    expect(result.explanations.some((entry) => entry.title.includes('unavailable'))).toBe(true)
  })

  it('reports missing push-up angles as unavailable instead of zero', () => {
    const result = generatePerformanceInsights(pushup({ minimum_elbow_angle_degrees: null, maximum_elbow_angle_degrees: null, average_elbow_angle_degrees: null }))
    expect(result.limitations.some((entry) => entry.title.includes('unavailable'))).toBe(true)
    expect(JSON.stringify(result)).not.toContain('0°')
  })

  it('does not claim broad-jump physical distance when calibration is absent', () => {
    const response = protocol('standing_broad_jump', { physical_distance_available: false, accepted_attempts: [{ horizontal_displacement_px: 80, physical_distance_available: false, distance_cm: null }] })
    const result = generatePerformanceInsights(response)
    expect(result.explanations.find((entry) => entry.title === 'Physical distance unavailable')?.detail).toContain('not converted to centimeters')
  })
})
