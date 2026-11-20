import type { AssessmentResponse } from '../types'
import type { DemoRole } from '../auth/sessionStore'

export interface StoredAssessment {
  id: string
  savedAt: string
  submittedBy: DemoRole
  response: AssessmentResponse
}

export const ASSESSMENT_STORAGE_KEY = 'athletelens.assessments.v1'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isStoredAssessment(value: unknown): value is StoredAssessment {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.savedAt !== 'string'
      || (value.submittedBy !== 'athlete' && value.submittedBy !== 'admin') || !isRecord(value.response)) return false
  return typeof value.response.exercise_type === 'string'
    && typeof value.response.processing_status === 'string'
    && typeof value.response.input_filename === 'string'
}

function createId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export function readAssessments(): StoredAssessment[] {
  try {
    const raw = localStorage.getItem(ASSESSMENT_STORAGE_KEY)
    if (!raw) return []
    const value: unknown = JSON.parse(raw)
    if (!Array.isArray(value)) return []
    return value.filter(isStoredAssessment).sort((a, b) => b.savedAt.localeCompare(a.savedAt))
  } catch {
    return []
  }
}

export function getAssessment(id: string): StoredAssessment | null {
  return readAssessments().find((assessment) => assessment.id === id) ?? null
}

/** Store only the real API response and lightweight local metadata, never the video itself. */
export function saveAssessment(response: AssessmentResponse, submittedBy: DemoRole): StoredAssessment | null {
  try {
    const assessment: StoredAssessment = {
      id: createId(),
      savedAt: new Date().toISOString(),
      submittedBy,
      response,
    }
    localStorage.setItem(ASSESSMENT_STORAGE_KEY, JSON.stringify([assessment, ...readAssessments()]))
    return assessment
  } catch {
    return null
  }
}
