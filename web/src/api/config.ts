const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()

/** Shared API origin/prefix. Defaults to Vite's same-origin /api proxy. */
export const API_BASE_URL = configuredBaseUrl || '/api'

export const API_ENDPOINTS = {
  health: `${API_BASE_URL}/v1/health`,
  assessments: {
    pushups: `${API_BASE_URL}/v1/assessments/pushups`,
    curlups: `${API_BASE_URL}/v1/assessments/curlups`,
    verticalJumps: `${API_BASE_URL}/v1/assessments/vertical-jumps`,
    broadJumps: `${API_BASE_URL}/v1/assessments/broad-jumps`,
  },
} as const
