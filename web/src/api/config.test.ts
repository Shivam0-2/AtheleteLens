import { describe, expect, it } from 'vitest'
import { API_ENDPOINTS, resolveApiUrl } from './config'

describe('LAN-safe API URL configuration', () => {
  it('keeps API requests on the same origin by default', () => {
    expect(API_ENDPOINTS.assessments.pushups).toMatch(/^\/api\/v1\//)
  })

  it('keeps root-relative annotated video URLs on the current frontend origin', () => {
    expect(resolveApiUrl('/api/v1/assessments/annotated-videos/abc')).toBe('/api/v1/assessments/annotated-videos/abc')
  })

  it('preserves an absolute annotated video URL when explicitly configured', () => {
    expect(resolveApiUrl('http://192.168.1.25:8000/api/v1/assessments/annotated-videos/abc'))
      .toBe('http://192.168.1.25:8000/api/v1/assessments/annotated-videos/abc')
  })
})
