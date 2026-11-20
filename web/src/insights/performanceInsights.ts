import type { AssessmentResponse } from '../types'

export interface InsightItem {
  title: string
  detail: string
  evidenceFields: string[]
}

export interface PerformanceInsights {
  strengths: InsightItem[]
  improvementAreas: InsightItem[]
  trainingFocus: InsightItem[]
  explanations: InsightItem[]
  limitations: InsightItem[]
}

type JsonRecord = Record<string, unknown>

function record(value: unknown): JsonRecord | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as JsonRecord : null
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function list(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(record).filter((item): item is JsonRecord => item !== null) : []
}

function item(title: string, detail: string, ...evidenceFields: string[]): InsightItem {
  return { title, detail, evidenceFields }
}

function format(value: number, digits = 1): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(digits)
}

function spreadOf(values: number[]): { min: number; max: number; spread: number } | null {
  if (values.length < 2) return null
  const min = Math.min(...values)
  const max = Math.max(...values)
  return { min, max, spread: max - min }
}

function commonQuality(response: AssessmentResponse, output: PerformanceInsights) {
  const pose = finiteNumber(response.pose_detection_percentage)
  const coverage = finiteNumber(response.measurement_coverage_percentage)
  if (pose !== null && pose < 100) {
    output.improvementAreas.push(item(
      'Some frames had no detected pose',
      `${format(pose)}% of frames had a detected pose. This is a tracking-quality observation, not a performance rating.`,
      'pose_detection_percentage', 'frames_with_detected_poses', 'total_frames_processed',
    ))
  }
  if (coverage !== null && coverage < 100) {
    const label = coverage === 0 ? 'No valid measurements were reported' : 'Some frames had no valid measurement'
    output.improvementAreas.push(item(
      label,
      `${format(coverage)}% measurement coverage was reported. Missing measurements are not treated as zero.`,
      'measurement_coverage_percentage', 'measurement_quality',
    ))
  }
  if (pose === null || coverage === null) {
    output.limitations.push(item(
      'Quality fields are incomplete',
      'Pose detection or measurement coverage was not supplied, so tracking quality cannot be summarized from this response.',
      'pose_detection_percentage', 'measurement_coverage_percentage',
    ))
  }
}

function addRejected(response: AssessmentResponse, rejected: JsonRecord[], output: PerformanceInsights) {
  if (rejected.length === 0) return
  const reasons = [...new Set(rejected.map((entry) => entry.reason).filter((reason): reason is string => typeof reason === 'string' && reason.length > 0))]
  output.improvementAreas.push(item(
    'Review rejected movement cycles',
    `${rejected.length} cycle${rejected.length === 1 ? ' was' : 's were'} rejected${reasons.length ? ` (${reasons.join(', ')})` : ''}. The event list below contains their timestamps and available reasons.`,
    response.exercise_type === 'push_up' ? 'rejected_events' : 'rejected_attempts',
  ))
}

function addDurationConsistency(events: JsonRecord[], field: string, label: string, output: PerformanceInsights) {
  const values = events.map((event) => finiteNumber(event.duration_seconds)).filter((value): value is number => value !== null)
  const spread = spreadOf(values)
  if (!spread) return
  output.explanations.push(item(
    `${label} duration spread`,
    `Reported accepted-cycle durations span ${format(spread.min, 2)} to ${format(spread.max, 2)} seconds (range ${format(spread.spread, 2)} seconds). This describes the recorded sample; it is not a validated consistency grade.`,
    field,
  ))
  output.trainingFocus.push(item(
    'Practice a comfortable, repeatable tempo',
    'As general, non-individualized guidance, controlled repetitions at a comfortable pace may help you focus on repeatability. The duration range above is descriptive and is not compared with a standard.',
    field,
  ))
}

export function generatePerformanceInsights(response: AssessmentResponse): PerformanceInsights {
  const output: PerformanceInsights = { strengths: [], improvementAreas: [], trainingFocus: [], explanations: [], limitations: [] }
  const summary = response.measurement_summary ?? {}
  const accepted = response.exercise_type === 'push_up'
    ? list(response.accepted_repetition_timestamps)
    : list(response.accepted_attempts)
  const rejected = response.exercise_type === 'push_up'
    ? list(response.rejected_events)
    : list(response.rejected_attempts)
  const acceptedCount = response.exercise_type === 'push_up'
    ? finiteNumber(response.detected_repetitions)
    : finiteNumber(response.accepted_attempt_count)
  const countLabel = response.exercise_type === 'push_up' || response.exercise_type === 'curl_up' ? 'repetitions' : 'attempts'

  if (acceptedCount !== null && acceptedCount > 0) {
    output.strengths.push(item(
      'Completed movement cycles were recorded',
      `The assessment recorded ${format(acceptedCount, 0)} accepted ${countLabel}. This reports detector output and does not compare it with a performance standard.`,
      response.exercise_type === 'push_up' ? 'detected_repetitions' : 'accepted_attempt_count',
      response.exercise_type === 'push_up' ? 'accepted_repetition_timestamps' : 'accepted_attempts',
    ))
  } else if (acceptedCount === 0) {
    output.improvementAreas.push(item(
      'No accepted cycles were reported',
      `This response contains zero accepted ${countLabel}; no cycle-specific performance measurement is available. Review the recorded rejected events and tracking indicators before drawing conclusions.`,
      response.exercise_type === 'push_up' ? 'detected_repetitions' : 'accepted_attempt_count',
      response.exercise_type === 'push_up' ? 'rejected_events' : 'rejected_attempts',
    ))
  } else {
    output.limitations.push(item('Accepted-cycle count unavailable', 'The response does not contain an accepted-cycle count, so a count-based insight cannot be generated.', response.exercise_type === 'push_up' ? 'detected_repetitions' : 'accepted_attempt_count'))
  }

  addRejected(response, rejected, output)

  if (response.exercise_type === 'push_up') {
    const min = finiteNumber(response.minimum_elbow_angle_degrees)
    const max = finiteNumber(response.maximum_elbow_angle_degrees)
    const avg = finiteNumber(response.average_elbow_angle_degrees)
    if (min !== null && max !== null) {
      output.explanations.push(item(
        'Elbow-angle range',
        `The response reports elbow angles from ${format(min)}° to ${format(max)}°${avg === null ? '' : `, with an average of ${format(avg)}°`}. These are measured values, not targets or form grades.`,
        'minimum_elbow_angle_degrees', 'maximum_elbow_angle_degrees', 'average_elbow_angle_degrees',
      ))
      if (acceptedCount !== null && acceptedCount > 0) output.strengths.push(item('Elbow-angle measurements were available', 'An angle range was returned for this video. It does not establish whether each repetition met a form standard.', 'minimum_elbow_angle_degrees', 'maximum_elbow_angle_degrees'))
    } else {
      output.limitations.push(item('Elbow-angle range unavailable', 'One or more aggregate elbow-angle values are missing; no angle range is inferred.', 'minimum_elbow_angle_degrees', 'maximum_elbow_angle_degrees', 'average_elbow_angle_degrees'))
    }
    addDurationConsistency(accepted, 'accepted_repetition_timestamps', 'Accepted repetition', output)
    output.explanations.push(item('How the push-up values are measured', 'Elbow angles are calculated from shoulder, elbow, and wrist landmarks. Repetition events come from the existing phase detector.', 'average_elbow_angle_degrees', 'minimum_elbow_angle_degrees', 'maximum_elbow_angle_degrees', 'accepted_repetition_timestamps'))
    output.limitations.push(item('No validated push-up reference standard is applied', 'The interface does not assign a grade or compare elbow angles, count, or duration with a normative benchmark.', 'detector_configuration'))
  } else if (response.exercise_type === 'curl_up') {
    const avg = finiteNumber(summary.average_included_angle_deg)
    if (avg !== null) {
      output.explanations.push(item('Average shoulder–hip–knee included angle', `The API reports an average included angle of ${format(avg)}°.`, 'measurement_summary.average_included_angle_deg'))
    } else {
      output.explanations.push(item('Torso–thigh angle unavailable', 'The response does not contain a numeric average included angle.', 'measurement_summary.average_included_angle_deg'))
    }
    output.limitations.push(item('Per-frame curl-up angle range is unavailable', 'The API response includes an average included angle, but its service omits the per-frame measurements. Minimum and maximum angles cannot be derived from this saved response.', 'measurement_summary.average_included_angle_deg'))
    addDurationConsistency(accepted, 'accepted_attempts', 'Accepted curl-up', output)
    output.explanations.push(item('How the curl-up values are measured', 'The included angle uses shoulder, hip, and knee landmarks. Configured thresholds define the experimental movement cycle; no standardized curl-up protocol is specified by the project.', 'measurement_summary.average_included_angle_deg', 'detector_configuration', 'accepted_attempts'))
    output.limitations.push(item('Curl-up protocol is experimental', 'No validated curl-up repetition standard or performance benchmark is applied.', 'detector_configuration', 'warnings'))
  } else if (response.exercise_type === 'vertical_jump') {
    const heights = accepted.map((event) => finiteNumber(event.flight_time_height_cm)).filter((value): value is number => value !== null)
    if (heights.length > 0) {
      const spread = spreadOf(heights)
      output.explanations.push(item(
        'Flight-time-derived height estimates',
        `The response contains ${heights.length} numeric per-attempt height estimate${heights.length === 1 ? '' : 's'}${spread ? `, ranging from ${format(spread.min, 2)} to ${format(spread.max, 2)} cm (range ${format(spread.spread, 2)} cm)` : ''}. These are experimental flight-time estimates.`,
        'accepted_attempts.flight_time_height_cm', 'measurement_summary.height_method',
      ))
        if (spread) output.trainingFocus.push(item('Use the recorded attempts to guide practice discussion', 'The variation is descriptive only. If practicing jumps, use a clear landing area and allow recovery between comfortable efforts; this is general guidance, not an individualized prescription.', 'accepted_attempts.flight_time_height_cm'))
    } else {
      output.explanations.push(item('Flight-time height unavailable', 'No accepted attempt contains a numeric flight_time_height_cm value; missing height is not treated as zero.', 'accepted_attempts.flight_time_height_cm'))
    }
    output.limitations.push(item('Height method and assumption', 'Height is estimated from flight time using h = g × t² / 8 and assumes equal center-of-mass height at takeoff and landing. Foot/ankle events are video proxies, not force-plate contact measurements.', 'accepted_attempts.flight_time_height_cm', 'measurement_summary.height_method', 'measurement_summary.height_assumption'))
  } else {
    const calibration = record(summary.calibration)
    const availableAttempts = accepted.filter((event) => event.physical_distance_available === true && finiteNumber(event.distance_cm) !== null)
    const distances = availableAttempts.map((event) => finiteNumber(event.distance_cm) as number)
    if (distances.length > 0) {
      const spread = spreadOf(distances)
      output.explanations.push(item(
        'Calibrated broad-jump distance',
        `Physical distance is available for ${distances.length} accepted attempt${distances.length === 1 ? '' : 's'}${spread ? `; the reported distances range from ${format(spread.min)} to ${format(spread.max)} cm` : ''}.`,
        'physical_distance_available', 'accepted_attempts.distance_cm', 'measurement_summary.calibration',
      ))
      if (spread) output.trainingFocus.push(item('Use repeatable, comfortable takeoff practice', 'Distance variation describes these attempts only; it is not a grade. General practice may focus on controlled, repeatable takeoffs with a safe landing area.', 'accepted_attempts.distance_cm'))
    } else {
      output.explanations.push(item('Physical distance unavailable', 'No accepted attempt contains an available calibrated distance. Pixel displacement, when present, remains a separate image measurement and is not converted to centimeters here.', 'physical_distance_available', 'accepted_attempts.horizontal_displacement_px', 'accepted_attempts.distance_cm', 'measurement_summary.calibration'))
      if (response.physical_distance_available !== true || calibration?.status !== 'valid') {
        output.improvementAreas.push(item('Calibration is needed for physical distance', 'For a physical distance measurement, provide a valid ground reference under the documented camera and same-ground-plane assumptions.', 'measurement_summary.calibration', 'physical_distance_available'))
      }
    }
    output.explanations.push(item('How broad-jump distance is measured', 'Heel displacement is reported in pixels. Meters and centimeters are available only when the response marks physical distance available and includes calibrated values.', 'accepted_attempts.horizontal_displacement_px', 'accepted_attempts.distance_m', 'accepted_attempts.distance_cm', 'accepted_attempts.physical_distance_available'))
    output.limitations.push(item('Calibration is local to the reference plane', 'A single scale does not correct perspective distortion across the image. Camera position and the athlete’s ground plane affect physical-distance interpretation.', 'measurement_summary.calibration', 'detector_configuration'))
  }

  commonQuality(response, output)
  if (output.trainingFocus.length === 0) {
    output.trainingFocus.push(item(
      'Keep practice comfortable and controlled',
      'General guidance only: practice the selected movement at a comfortable effort, prioritize a safe environment, and stop if you feel pain or unusual discomfort. This is not individualized coaching.',
      response.exercise_type === 'push_up' ? 'detected_repetitions' : 'accepted_attempt_count',
    ))
  }
  if (response.assessment_notice) output.limitations.push(item('Preliminary assessment notice', response.assessment_notice, 'assessment_notice'))
  return output
}
