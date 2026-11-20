import { afterEach, describe, expect, it, vi } from 'vitest'
import { getAssessment, readAssessments, saveAssessment } from './assessmentStore'
import type { AssessmentResponse } from '../types'

function memoryStorage() {
  const values = new Map<string, string>()
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value) },
    removeItem: (key: string) => { values.delete(key) },
  }
}

const response = {
  status: 'preliminary_unvalidated', processing_status: 'completed', input_filename: 'real-video.mp4',
  exercise_type: 'push_up', video: { fps: 30, width: 640, height: 480 },
  total_frames_processed: 5, frames_with_detected_poses: 5, frames_with_valid_elbow_angle: 4,
  pose_detection_percentage: 100, detected_repetitions: 0,
  average_elbow_angle_degrees: null, minimum_elbow_angle_degrees: null, maximum_elbow_angle_degrees: null,
  accepted_repetition_timestamps: [], rejected_events: [],
} satisfies AssessmentResponse

afterEach(() => vi.unstubAllGlobals())

describe('local assessment history', () => {
  it('stores and retrieves only the supplied API response', () => {
    vi.stubGlobal('localStorage', memoryStorage())
    const saved = saveAssessment(response, 'athlete')
    expect(saved?.response).toEqual(response)
    expect(readAssessments()).toHaveLength(1)
    expect(getAssessment(saved!.id)).toEqual(saved)
  })

  it('returns an empty history and reports storage failure without inventing a record', () => {
    vi.stubGlobal('localStorage', { getItem: () => null, setItem: () => { throw new Error('quota') } })
    expect(readAssessments()).toEqual([])
    expect(saveAssessment(response, 'athlete')).toBeNull()
  })
})
