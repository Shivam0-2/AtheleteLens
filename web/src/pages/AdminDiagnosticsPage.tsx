import { useEffect, useState } from 'react'
import { Activity, Braces, CircleAlert, FileClock, Ruler, Server } from 'lucide-react'
import { API_BASE_URL, API_ENDPOINTS } from '../api/config'
import { AttemptList } from '../components/AttemptList'
import { QualitySummary } from '../components/QualitySummary'
import { Warnings } from '../components/Warnings'
import { exercises } from '../data/exercises'
import { readAssessments, type StoredAssessment } from '../data/assessmentStore'
import type { AssessmentResponse } from '../types'

function labelFor(response: AssessmentResponse): string {
  const lookup: Record<string, string> = { push_up: 'pushups', curl_up: 'curlups', vertical_jump: 'vertical-jumps', standing_broad_jump: 'broad-jumps' }
  return exercises.find((item) => item.key === lookup[response.exercise_type])?.name ?? response.exercise_type
}

function accepted(response: AssessmentResponse) {
  return response.exercise_type === 'push_up' ? response.accepted_repetition_timestamps : response.accepted_attempts
}

function rejected(response: AssessmentResponse) {
  return response.exercise_type === 'push_up' ? response.rejected_events : response.rejected_attempts
}

function acceptedCount(response: AssessmentResponse) {
  return response.exercise_type === 'push_up' ? response.detected_repetitions : response.accepted_attempt_count
}

function calibrationOf(record: StoredAssessment): Record<string, unknown> | null {
  if (record.response.exercise_type !== 'standing_broad_jump') return null
  const value = record.response.measurement_summary.calibration
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : null
}

function shown(value: unknown): string {
  if (value === null || value === undefined) return 'Unavailable'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(2)
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  return String(value)
}

function SubmissionCard({ item }: { item: StoredAssessment }) {
  const response = item.response
  const calibration = calibrationOf(item)
  return <article className="admin-submission-card">
    <div className="admin-record-heading"><div><span className="eyebrow">{labelFor(response)} · {item.submittedBy.toUpperCase()} DEMO SESSION</span><h3>{response.input_filename}</h3><p>{new Date(item.savedAt).toLocaleString()}</p></div><span className="status-tag status-config">{response.processing_status}</span></div>
    <div className="admin-record-grid">
      <div><small>Video metadata</small><strong>{response.video.width} × {response.video.height} px · {shown(response.video.fps)} FPS</strong></div>
      <div><small>Pose detection</small><strong>{shown(response.pose_detection_percentage)}%</strong></div>
      <div><small>Measurement coverage</small><strong>{shown(response.measurement_coverage_percentage)}{response.measurement_coverage_percentage === undefined ? '' : '%'}</strong></div>
      <div><small>Accepted / rejected</small><strong>{acceptedCount(response)} / {rejected(response).length}</strong></div>
    </div>
    <div className="admin-event-quality"><div className="content-panel"><div className="panel-heading"><div><span className="eyebrow">EVENT REVIEW</span><h3>Accepted and rejected</h3></div></div><AttemptList accepted={accepted(response)} rejected={rejected(response)} exerciseType={response.exercise_type} /></div><div className="content-panel"><div className="panel-heading"><div><span className="eyebrow">QUALITY</span><h3>Tracking summary</h3></div></div><QualitySummary response={response} /><details className="technical-details"><summary>Quality fields</summary><pre>{JSON.stringify(response.measurement_quality ?? {}, null, 2)}</pre></details></div></div>
    {calibration && <section className="admin-calibration"><div className="admin-section-title"><Ruler size={15} /><strong>Broad-jump calibration</strong></div><div className="admin-record-grid"><div><small>Status</small><strong>{shown(calibration.status)}</strong></div><div><small>Physical distance</small><strong>{shown(calibration.physical_distance_available)}</strong></div><div><small>Scale</small><strong>{calibration.pixels_per_meter == null ? 'Unavailable' : `${shown(calibration.pixels_per_meter)} px/m`}</strong></div><div><small>Reference length</small><strong>{calibration.reference_length_m == null ? 'Unavailable' : `${shown(calibration.reference_length_m)} m`}</strong></div></div><p>{shown(calibration.explanation)}</p><p className="perspective-note">{shown(calibration.assumptions)}</p></section>}
    <Warnings items={response.warnings ?? []} />
    <details className="technical-details raw-response"><summary><Braces size={14} /> Raw API response JSON</summary><pre>{JSON.stringify(response, null, 2)}</pre></details>
  </article>
}

export function AdminDiagnosticsPage() {
  const [records, setRecords] = useState(() => readAssessments())
  useEffect(() => {
    const refresh = () => setRecords(readAssessments())
    window.addEventListener('storage', refresh)
    return () => window.removeEventListener('storage', refresh)
  }, [])

  return <div className="page-stack"><section className="page-intro"><span className="eyebrow">LOCAL DEMO STORAGE</span><h2>Assessment diagnostics</h2><p>Inspect only real API responses saved in this browser. No server log, user directory, or assessment record is generated by this page.</p></section>
    <div className="diagnostics-grid"><article className="diagnostic-card"><div className="diagnostic-card-top"><div className="diag-icon"><Server size={18} /></div><span className="status-tag status-config">CONFIGURED</span></div><span className="eyebrow">API BASE URL</span><strong className="diag-value">{API_BASE_URL}</strong><p>Health route: <code>{API_ENDPOINTS.health}</code></p></article><article className="diagnostic-card"><div className="diagnostic-card-top"><div className="diag-icon"><Activity size={18} /></div><span className="status-tag">NOT CHECKED</span></div><span className="eyebrow">SAVED SUBMISSIONS</span><strong className="diag-value">{records.length}</strong><p>Actual assessment responses stored in this browser.</p></article><article className="diagnostic-card"><div className="diagnostic-card-top"><div className="diag-icon"><Braces size={18} /></div><span className="status-tag">4 ROUTES</span></div><span className="eyebrow">ASSESSMENT API</span><strong className="diag-value">Four protocols</strong><p>Push-up, curl-up, vertical jump, and broad jump.</p></article></div>
    <div className="admin-history-heading"><div><span className="eyebrow">BROWSER-LOCAL RECORDS</span><h2>Submissions</h2></div><FileClock size={19} /></div>
    {records.length ? <div className="admin-submission-list">{records.map((item) => <SubmissionCard key={item.id} item={item} />)}</div> : <div className="history-empty admin-empty"><CircleAlert size={18} /><span>No assessment responses are saved in this browser yet. Submit a real video as the athlete demo role, then return here.</span></div>}
    <div className="warnings-panel"><div className="warning-heading"><CircleAlert size={17} /><strong>Demo limitations</strong></div><ul><li>Role checks and credentials are client-side demo fixtures, not production security.</li><li>Records are stored only in this browser and are not shared with the backend.</li><li>Unavailable response fields remain unavailable; zero values are shown as returned.</li></ul></div>
  </div>
}
