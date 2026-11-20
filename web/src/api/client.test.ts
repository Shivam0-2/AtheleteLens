import { afterEach, describe, expect, it, vi } from 'vitest'
import { API_ENDPOINTS } from './config'
import { AssessmentApiError, submitAssessment } from './client'

function protocolResponse(exerciseType = 'curl_up') {
  return {
    status: 'preliminary_unvalidated', processing_status: 'completed',
    input_filename: 'sample.mp4', exercise_type: exerciseType,
    video: { fps: 30, width: 640, height: 480 },
    total_frames_processed: 10, frames_with_detected_poses: 9,
    frames_with_valid_measurements: 8, pose_detection_percentage: 90,
    measurement_coverage_percentage: 80, measurement_quality: {},
    accepted_attempt_count: 1, accepted_attempts: [], rejected_attempt_count: 0,
    rejected_attempts: [], measurement_summary: {}, physical_distance_attempt_count: null,
    physical_distance_available: null, detector_configuration: {},
    annotated_video_available: false, annotated_video_path: null, warnings: [],
  }
}

function pushupResponse() {
  return {
    status: 'preliminary_unvalidated', processing_status: 'completed',
    input_filename: 'sample.mp4', exercise_type: 'push_up',
    video: { fps: 30, width: 640, height: 480 }, total_frames_processed: 10,
    frames_with_detected_poses: 9, frames_with_valid_elbow_angle: 8,
    pose_detection_percentage: 90, detected_repetitions: 1,
    average_elbow_angle_degrees: null, minimum_elbow_angle_degrees: null,
    maximum_elbow_angle_degrees: null, accepted_repetition_timestamps: [],
    rejected_events: [],
  }
}

function videoFile() {
  return new File(['video bytes'], 'sample.mp4', { type: 'video/mp4' })
}

afterEach(() => vi.restoreAllMocks())

describe('assessment API client', () => {
  it.each([
    ['pushups', API_ENDPOINTS.assessments.pushups, pushupResponse(), undefined],
    ['curlups', API_ENDPOINTS.assessments.curlups, protocolResponse('curl_up'), { extended_angle_deg: 115, flexed_angle_deg: 100 }],
    ['vertical-jumps', API_ENDPOINTS.assessments.verticalJumps, protocolResponse('vertical_jump'), undefined],
    ['broad-jumps', API_ENDPOINTS.assessments.broadJumps, protocolResponse('standing_broad_jump'), undefined],
  ] as const)('posts video to %s with the matching response contract', async (exercise, endpoint, body, config) => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)

    const result = await submitAssessment(exercise, videoFile(), config)

    expect(fetchMock).toHaveBeenCalledOnce()
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe(endpoint)
    expect(init.method).toBe('POST')
    expect(init.headers).toBeUndefined()
    expect(init.body).toBeInstanceOf(FormData)
    expect((init.body as FormData).get('video')).toBeInstanceOf(File)
    expect((init.body as FormData).has('config_json')).toBe(config !== undefined)
    expect(result.exercise_type).toBe(body.exercise_type)
  })

  it('sends curl-up thresholds and broad-jump calibration as JSON form fields', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(protocolResponse('curl_up')), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(protocolResponse('standing_broad_jump')), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const curlConfig = { extended_angle_deg: 115, flexed_angle_deg: 100 }
    const broadConfig = {
      reference_endpoint_a_x_px: 12, reference_endpoint_a_y_px: 200,
      reference_endpoint_b_x_px: 212, reference_endpoint_b_y_px: 200,
      reference_length_m: 2,
    }

    await submitAssessment('curlups', videoFile(), curlConfig)
    await submitAssessment('broad-jumps', videoFile(), broadConfig)

    const curlForm = fetchMock.mock.calls[0][1]?.body as FormData
    const broadForm = fetchMock.mock.calls[1][1]?.body as FormData
    expect(JSON.parse(String(curlForm.get('config_json')))).toEqual(curlConfig)
    expect(JSON.parse(String(broadForm.get('config_json')))).toEqual(broadConfig)
  })

  it('reports backend validation errors and handles non-JSON error bodies', async () => {
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: [{ loc: ['extended_angle_deg'], msg: 'invalid threshold' }] }), { status: 422 }))
      .mockResolvedValueOnce(new Response('service unavailable', { status: 503 })))

    await expect(submitAssessment('curlups', videoFile(), { extended_angle_deg: 150, flexed_angle_deg: 100 }))
      .rejects.toMatchObject({ name: 'AssessmentApiError', status: 422, message: 'extended_angle_deg: invalid threshold' })
    await expect(submitAssessment('pushups', videoFile()))
      .rejects.toBeInstanceOf(AssessmentApiError)
  })

  it('rejects malformed successful responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"processing_status":"completed"}', { status: 200 })))
    await expect(submitAssessment('pushups', videoFile()))
      .rejects.toMatchObject({ name: 'AssessmentApiError', message: 'The assessment server response did not match the expected API schema.' })
  })
})
