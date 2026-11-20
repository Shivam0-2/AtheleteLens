import { API_ENDPOINTS } from './config'
import type { AssessmentResponse, ExerciseKey } from '../types'

export class AssessmentApiError extends Error {
  readonly status?: number
  readonly detail?: unknown

  constructor(message: string, status?: number, detail?: unknown) {
    super(message)
    this.name = 'AssessmentApiError'
    this.status = status
    this.detail = detail
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string')
}

/** Runtime-check the fields shared by both Pydantic response contracts. */
export function isAssessmentResponse(value: unknown): value is AssessmentResponse {
  if (!isRecord(value) || typeof value.processing_status !== 'string'
      || typeof value.input_filename !== 'string' || typeof value.exercise_type !== 'string'
      || !isRecord(value.video) || !isNumber(value.video.fps)
      || !isNumber(value.video.width) || !isNumber(value.video.height)
      || !isNumber(value.total_frames_processed) || !isNumber(value.frames_with_detected_poses)
      || !isNumber(value.pose_detection_percentage)) return false

  if (value.warnings !== undefined && !isStringArray(value.warnings)) return false

  if (value.exercise_type === 'push_up') {
    return isNumber(value.detected_repetitions)
      && isNumber(value.frames_with_valid_elbow_angle)
      && (value.average_elbow_angle_degrees === null || isNumber(value.average_elbow_angle_degrees))
      && (value.minimum_elbow_angle_degrees === null || isNumber(value.minimum_elbow_angle_degrees))
      && (value.maximum_elbow_angle_degrees === null || isNumber(value.maximum_elbow_angle_degrees))
      && Array.isArray(value.accepted_repetition_timestamps)
      && Array.isArray(value.rejected_events)
  }

  return (value.exercise_type === 'curl_up' || value.exercise_type === 'vertical_jump'
      || value.exercise_type === 'standing_broad_jump')
    && isNumber(value.frames_with_valid_measurements)
    && isNumber(value.measurement_coverage_percentage)
    && isRecord(value.measurement_quality)
    && isNumber(value.accepted_attempt_count)
    && Array.isArray(value.accepted_attempts)
    && isNumber(value.rejected_attempt_count)
    && Array.isArray(value.rejected_attempts)
    && isRecord(value.measurement_summary)
    && isRecord(value.detector_configuration)
    && typeof value.annotated_video_available === 'boolean'
    && (typeof value.annotated_video_path === 'string' || value.annotated_video_path === null)
      && (value.annotated_video_url === undefined || typeof value.annotated_video_url === 'string' || value.annotated_video_url === null)
}

function describeErrorDetail(detail: unknown): string | null {
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) {
    const messages = detail.flatMap((entry) => {
      if (!isRecord(entry)) return []
      const message = typeof entry.msg === 'string' ? entry.msg : null
      const location = Array.isArray(entry.loc) ? entry.loc.map(String).join(' → ') : ''
      return message ? [`${location ? `${location}: ` : ''}${message}`] : []
    })
    if (messages.length) return messages.join('; ')
  }
  return null
}

async function readError(response: Response): Promise<AssessmentApiError> {
  const text = await response.text().catch(() => '')
  let body: unknown
  if (text) {
    try { body = JSON.parse(text) as unknown } catch { body = text }
  }
  const detail = isRecord(body) ? body.detail : body
  const detailMessage = describeErrorDetail(detail)
  const statusMessage: Record<number, string> = {
    413: 'The video exceeds the server upload limit.',
    415: 'This video format is not supported by the assessment service.',
    422: 'The video or assessment configuration was not accepted.',
    503: 'The pose model is unavailable on the assessment server.',
  }
  const message = detailMessage ?? statusMessage[response.status]
    ?? `Assessment request failed with HTTP ${response.status}.`
  return new AssessmentApiError(message, response.status, detail)
}

/** Upload one video and optional exercise configuration to its existing API route. */
export async function submitAssessment(
  exercise: ExerciseKey,
  video: File,
  config?: Record<string, unknown>,
): Promise<AssessmentResponse> {
  if (exercise === 'curlups') {
    const extended = config?.extended_angle_deg
    const flexed = config?.flexed_angle_deg
    if (typeof extended !== 'number' || typeof flexed !== 'number'
        || !Number.isFinite(extended) || !Number.isFinite(flexed)
        || flexed < 0 || extended > 180 || flexed >= extended) {
      throw new AssessmentApiError('Curl-up requires valid extended and flexed angle thresholds (0 ≤ flexed < extended ≤ 180 degrees).')
    }
  }

  const form = new FormData()
  form.append('video', video, video.name)
  if (config !== undefined) form.append('config_json', JSON.stringify(config))

  const endpointKey = {
    pushups: 'pushups',
    curlups: 'curlups',
    'vertical-jumps': 'verticalJumps',
    'broad-jumps': 'broadJumps',
  } as const
  let response: Response
  try {
    response = await fetch(API_ENDPOINTS.assessments[endpointKey[exercise]], {
      method: 'POST',
      body: form,
    })
  } catch (error) {
    throw new AssessmentApiError(
      error instanceof Error ? `Could not reach the assessment API: ${error.message}` : 'Could not reach the assessment API.',
    )
  }

  if (!response.ok) throw await readError(response)

  let payload: unknown
  try {
    payload = await response.json() as unknown
  } catch {
    throw new AssessmentApiError('The assessment server returned an invalid JSON response.', response.status)
  }
  if (!isAssessmentResponse(payload)) {
    throw new AssessmentApiError('The assessment server response did not match the expected API schema.', response.status, payload)
  }
  const expectedType = {
    pushups: 'push_up',
    curlups: 'curl_up',
    'vertical-jumps': 'vertical_jump',
    'broad-jumps': 'standing_broad_jump',
  } as const
  if (payload.exercise_type !== expectedType[exercise]) {
    throw new AssessmentApiError('The assessment server returned a response for a different exercise.', response.status, payload)
  }
  return payload
}
