const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim().replace(/\/+$/, '')

/** Shared API origin/prefix. Defaults to Vite's same-origin /api proxy. */
export const API_BASE_URL = configuredBaseUrl || '/api'

/** Resolve API-returned root-relative media URLs against the configured API origin. */
export function resolveApiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path
  if (API_BASE_URL.startsWith('/')) return path
  try {
    return new URL(path, API_BASE_URL).toString()
  } catch {
    return path
  }
}

export const API_ENDPOINTS = {
  health: `${API_BASE_URL}/v1/health`,
  assessments: {
    pushups: `${API_BASE_URL}/v1/assessments/pushups`,
    curlups: `${API_BASE_URL}/v1/assessments/curlups`,
    verticalJumps: `${API_BASE_URL}/v1/assessments/vertical-jumps`,
    broadJumps: `${API_BASE_URL}/v1/assessments/broad-jumps`,
    history: `${API_BASE_URL}/v1/assessments/history`,
    adminHistory: `${API_BASE_URL}/v1/assessments/history/admin`,
  },
} as const
