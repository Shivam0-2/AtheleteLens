import { API_ENDPOINTS } from '../api/config'
import type { ExerciseInfo, ExerciseKey } from '../types'

export const exercises: ExerciseInfo[] = [
  {
    key: 'pushups', name: 'Push-ups', shortName: 'Push-up',
    description: 'Track elbow movement through a controlled repetition cycle.',
    focus: 'Upper body · Repetition count', icon: '↗', endpoint: API_ENDPOINTS.assessments.pushups,
  },
  {
    key: 'curlups', name: 'Curl-ups', shortName: 'Curl-up',
    description: 'Review torso movement and repetition-level measurements.',
    focus: 'Core · Repetition count', icon: '◒', endpoint: API_ENDPOINTS.assessments.curlups,
  },
  {
    key: 'vertical-jumps', name: 'Standing vertical jump', shortName: 'Vertical jump',
    description: 'Estimate jump attempts from takeoff, flight, and landing events.',
    focus: 'Lower body · Flight-time estimate', icon: '↑', endpoint: API_ENDPOINTS.assessments.verticalJumps,
  },
  {
    key: 'broad-jumps', name: 'Standing broad jump', shortName: 'Broad jump',
    description: 'Measure horizontal displacement and optionally calibrate distance.',
    focus: 'Lower body · Ground displacement', icon: '↔', endpoint: API_ENDPOINTS.assessments.broadJumps,
  },
]

export function getExercise(key: string | undefined): ExerciseInfo | undefined {
  return exercises.find((exercise) => exercise.key === key as ExerciseKey)
}
