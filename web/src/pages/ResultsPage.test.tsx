import { describe, expect, it } from 'vitest'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { DemoAuthProvider } from '../auth/AuthContext'
import { ResultsPage } from './ResultsPage'
import type { PushUpAssessmentResponse } from '../types'

const response: PushUpAssessmentResponse = {
  processing_status: 'completed', input_filename: 'pushup.mp4', exercise_type: 'push_up',
  video: { fps: 24, width: 1920, height: 1080 }, total_frames_processed: 325,
  frames_with_detected_poses: 325, frames_with_valid_elbow_angle: 320,
  measurement_coverage_percentage: 98.46, pose_detection_percentage: 100,
  detected_repetitions: 4, average_elbow_angle_degrees: 111,
  minimum_elbow_angle_degrees: 72, maximum_elbow_angle_degrees: 164,
  accepted_repetition_timestamps: [], rejected_events: [],
  measurement_summary: {}, annotated_video_available: true,
  annotated_video_url: '/api/v1/assessments/annotated-videos/0123456789abcdef0123456789abcdef',
  annotated_video_path: null, warnings: [],
}

function render(responseValue: PushUpAssessmentResponse) {
  return renderToStaticMarkup(createElement(
    MemoryRouter,
    { initialEntries: [{ pathname: '/results', state: { response: responseValue } }] },
    createElement(DemoAuthProvider, null,
      createElement(Routes, null,
        createElement(Route, { path: '/results', element: createElement(ResultsPage) }),
      ),
    ),
  ))
}

describe('ResultsPage annotated video playback', () => {
  it('renders the verified API retrieval URL in a native MP4 player', () => {
    const html = render(response)
    expect(html).toContain('<video')
    expect(html).toContain('controls')
    expect(html).toMatch(/playsinline/i)
    expect(html).toContain('src="/api/v1/assessments/annotated-videos/0123456789abcdef0123456789abcdef"')
    expect(html).toContain('type="video/mp4"')
  })

  it('shows an unavailable state instead of a player when no output exists', () => {
    const html = render({ ...response, annotated_video_available: false, annotated_video_url: null })
    expect(html).not.toContain('<video')
    expect(html).toContain('Annotated video unavailable for this response')
  })
})
