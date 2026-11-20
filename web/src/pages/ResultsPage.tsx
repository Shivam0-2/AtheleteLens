import { ArrowLeft, BookOpen, CheckCircle2, Clock3, FileSearch, Gauge, Info, Ruler, ScanLine, Sparkles, Target, TrendingUp } from 'lucide-react'
import { Link, useLocation, useParams } from 'react-router-dom'
import type { ReactNode } from 'react'
import { useDemoAuth } from '../auth/AuthContext'
import { AttemptList } from '../components/AttemptList'
import { QualitySummary } from '../components/QualitySummary'
import { Warnings } from '../components/Warnings'
import { getExercise } from '../data/exercises'
import { getAssessment, readAssessments } from '../data/assessmentStore'
import { generatePerformanceInsights } from '../insights/performanceInsights'
import type { AssessmentResponse } from '../types'

interface ResultsLocationState { response?: AssessmentResponse; exerciseKey?: string; recordId?: string; persistenceUnavailable?: boolean }

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown> : null
}

function prettyLabel(key: string): string {
  return key.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function formatted(value: unknown): string {
  if (value === null || value === undefined) return 'Unavailable'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(2)
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'string') return value.replaceAll('_', ' ')
  return 'See details'
}

function Metric({ label, value, suffix, detail }: { label: string; value: string | number; suffix?: string; detail?: string }) {
  return <article className="result-metric"><span>{label}</span><strong>{value}<small>{suffix}</small></strong>{detail && <small className="metric-detail">{detail}</small>}</article>
}

export function ResultsPage() {
  const location = useLocation()
  const { recordId: routeRecordId } = useParams()
  const { session } = useDemoAuth()
  const state = location.state as ResultsLocationState | null
  const stored = routeRecordId
    ? getAssessment(routeRecordId)
    : state?.recordId
      ? getAssessment(state.recordId)
      : state?.response
        ? null
        : readAssessments().find((item) => item.submittedBy === session?.role) ?? null
  const response = state?.response ?? stored?.response

  if (!response) return <div className="page-stack">
    <section className="page-intro"><span className="eyebrow">ASSESSMENT OUTPUT</span><h2>Results</h2><p>Results display the data returned by the FastAPI assessment service.</p></section>
    <div className="empty-results"><div className="empty-results-icon"><FileSearch size={27} /></div><h3>No assessment response available</h3><p>No saved assessment response was found in this browser. Submit a video to view its real API response here.</p><Link to="/exercises" className="button button-primary">Choose an assessment</Link></div>
  </div>

  const exerciseKeyForType: Record<string, string> = { push_up: 'pushups', curl_up: 'curlups', vertical_jump: 'vertical-jumps', standing_broad_jump: 'broad-jumps' }
  const insights = generatePerformanceInsights(response)
  const exercise = getExercise(state?.exerciseKey ?? exerciseKeyForType[response.exercise_type])
  const summary = response.exercise_type === 'push_up' ? response.measurement_summary ?? {} : response.measurement_summary
  const calibration = record(summary.calibration)
  const accepted = response.exercise_type === 'push_up' ? response.accepted_repetition_timestamps : response.accepted_attempts
  const rejected = response.exercise_type === 'push_up' ? response.rejected_events : response.rejected_attempts
  const acceptedCount = response.exercise_type === 'push_up' ? response.detected_repetitions : response.accepted_attempt_count
  const isVertical = response.exercise_type === 'vertical_jump'
  const isBroad = response.exercise_type === 'standing_broad_jump'
  const measurementDetail = response.exercise_type === 'push_up'
    ? `${response.frames_with_valid_elbow_angle} valid elbow angles`
    : `${response.frames_with_valid_measurements} valid measurements`
  const calibratedBroadAttempt = response.exercise_type === 'standing_broad_jump'
    ? response.accepted_attempts.find((event) => event.physical_distance_available === true)
    : undefined
  const summaryFields = Object.entries(summary).filter(([key, value]) => {
    if (['calibration', 'measurements', 'accepted_attempts', 'rejected_attempts', 'accepted_repetitions', 'attempts_with_physical_distance'].includes(key)) return false
    if (key.endsWith('_percentage') || key === 'mean_visibility' || key === 'detected_repetitions') return false
    return typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean' || value === null
  })
  const verticalExamples = isVertical ? (response.exercise_type === 'vertical_jump' ? response.accepted_attempts : []) : []
  const maxVerticalHeight = verticalExamples.reduce<number | null>((highest, attempt) => {
    const height = attempt.flight_time_height_cm
    return typeof height === 'number' && (highest === null || height > highest) ? height : highest
  }, null)

  return <div className="page-stack">
    <Link to={`/assessments/${state?.exerciseKey ?? exerciseKeyForType[response.exercise_type] ?? 'pushups'}`} className="back-link"><ArrowLeft size={15} /> Run another assessment</Link>
    <section className="results-heading"><div><span className="eyebrow">ACTUAL API RESPONSE · {response.input_filename}</span><h2>{exercise?.name ?? prettyLabel(response.exercise_type)}</h2><p>Processing status: <strong>{prettyLabel(response.processing_status)}</strong> · {response.video.width} × {response.video.height} px · {response.video.fps.toFixed(2)} FPS{stored ? ` · Saved ${new Date(stored.savedAt).toLocaleString()}` : ''}</p></div><span className="result-status"><CheckCircle2 size={15} /> {prettyLabel(response.processing_status)}</span></section>
    {state?.persistenceUnavailable && <div className="inline-note">This result is available for this page visit, but browser storage could not save it for later.</div>}
    <div className="preliminary-banner"><ScanLine size={16} /><span>{response.assessment_notice ?? 'Experimental computer-vision output; not scientifically validated.'}</span></div>

    <section className="result-metrics-grid">
      <Metric label={response.exercise_type === 'push_up' || response.exercise_type === 'curl_up' ? 'Accepted repetitions' : 'Accepted attempts'} value={acceptedCount} detail={`${rejected.length} rejected`} />
      <Metric label="Pose detection" value={response.pose_detection_percentage.toFixed(1)} suffix="%" detail={`${response.frames_with_detected_poses} / ${response.total_frames_processed} frames`} />
      <Metric label="Measurement coverage" value={response.measurement_coverage_percentage === undefined ? 'Unavailable' : response.measurement_coverage_percentage.toFixed(1)} suffix={response.measurement_coverage_percentage === undefined ? '' : '%'} detail={measurementDetail} />
      {response.exercise_type === 'push_up' && <Metric label="Average elbow angle" value={response.average_elbow_angle_degrees ?? 'Unavailable'} suffix={response.average_elbow_angle_degrees === null ? '' : '°'} detail={`Range ${formatted(response.minimum_elbow_angle_degrees)}° to ${formatted(response.maximum_elbow_angle_degrees)}°`} />}
      {response.exercise_type === 'curl_up' && <Metric label="Average included angle" value={formatted(summary.average_included_angle_deg)} suffix={typeof summary.average_included_angle_deg === 'number' ? '°' : ''} detail="Shoulder–hip–knee angle" />}
      {isVertical && <Metric label="Highest flight-time height" value={maxVerticalHeight === null ? 'Unavailable' : maxVerticalHeight.toFixed(2)} suffix={maxVerticalHeight === null ? '' : ' cm'} detail="Experimental estimate · equal takeoff/landing height assumed" />}
      {isBroad && <Metric label="Physical distance" value={response.physical_distance_available && typeof calibratedBroadAttempt?.distance_cm === 'number' ? calibratedBroadAttempt.distance_cm.toFixed(1) : 'Unavailable'} suffix={response.physical_distance_available && typeof calibratedBroadAttempt?.distance_cm === 'number' ? ' cm' : ''} detail={response.physical_distance_available ? `${response.physical_distance_attempt_count ?? 0} attempts with calibration` : 'Distance requires valid calibration'} />}
    </section>

    <section className="performance-insights" aria-labelledby="performance-insights-heading">
      <div className="insights-heading"><div><span className="eyebrow">EXPLAINABLE SUMMARY</span><h3 id="performance-insights-heading">Performance insights</h3><p>Exercise-specific observations derived only from fields in this assessment response. These are not validated grades.</p></div><Sparkles size={19} /></div>
      <div className="insight-grid">
        <InsightGroup icon={<TrendingUp size={16} />} title="What went well" items={insights.strengths} empty="No positive movement observation can be made from the available fields." />
        <InsightGroup icon={<Info size={16} />} title="Areas to improve" items={insights.improvementAreas} empty="No specific improvement area was identified from this response." />
        <InsightGroup icon={<Target size={16} />} title="Recommended training focus" items={insights.trainingFocus} empty="No training focus is available." />
        <InsightGroup icon={<BookOpen size={16} />} title="Measurement explanations" items={insights.explanations} empty="No exercise-specific measurement was reported." />
      </div>
      <InsightGroup icon={<Info size={16} />} title="Assessment limitations" items={insights.limitations} empty="No additional limitation text was returned." wide />
      <p className="insight-evidence-note">Each observation lists its source response fields. Tracking quality describes measurement availability and is separate from athletic performance.</p>
    </section>

    <div className="result-columns"><section className="content-panel"><div className="panel-heading"><div><span className="eyebrow">MOVEMENT EVENTS</span><h3>Accepted and rejected</h3></div><Clock3 size={17} /></div><AttemptList accepted={accepted} rejected={rejected} exerciseType={response.exercise_type} /></section>
      <section className="content-panel"><div className="panel-heading"><div><span className="eyebrow">TRACKING QUALITY</span><h3>Frame coverage</h3></div><Gauge size={17} /></div><QualitySummary response={response} />
        {response.measurement_quality && <details className="technical-details"><summary>Measurement quality details</summary><pre>{JSON.stringify(response.measurement_quality, null, 2)}</pre></details>}
      </section></div>

    {isBroad && calibration && <section className="content-panel calibration-result"><div className="panel-heading"><div><span className="eyebrow">GROUND REFERENCE</span><h3>Calibration</h3></div><Ruler size={17} /></div><div className="calibration-status-row"><span className={`status-tag ${calibration.status === 'valid' ? 'status-config' : ''}`}>{formatted(calibration.status)}</span><span>{calibration.physical_distance_available ? 'Physical distance available' : 'Physical distance unavailable'}</span></div><p>{formatted(calibration.explanation)}</p><dl>{[['Scale', calibration.pixels_per_meter == null ? null : `${formatted(calibration.pixels_per_meter)} px/m`], ['Reference length', calibration.reference_length_m == null ? null : `${formatted(calibration.reference_length_m)} m`], ['Scope', calibration.assumptions]].map(([label, value]) => <div key={String(label)}><dt>{String(label)}</dt><dd>{formatted(value)}</dd></div>)}</dl></section>}

    {summaryFields.length > 0 && <section className="content-panel"><div className="panel-heading"><div><span className="eyebrow">EXERCISE MEASUREMENTS</span><h3>Reported summary</h3></div></div><div className="summary-value-grid">{summaryFields.map(([key, value]) => <div key={key}><span>{prettyLabel(key)}</span><strong>{formatted(value)}{key.endsWith('_deg') || key.endsWith('_angle_deg') ? '°' : key.endsWith('_cm') ? ' cm' : key.endsWith('_m') ? ' m' : ''}</strong></div>)}</div></section>}

    {isVertical && <div className="method-note"><strong>Flight-time height estimate</strong><p>Reported height uses flight time and assumes equal center-of-mass height at takeoff and confirmed landing. Foot/ankle landmark contact timestamps are video proxies, not force-plate measurements.</p></div>}
    {!response.annotated_video_available && <div className="inline-note"><FileSearch size={15} /><span>The API did not generate an annotated video for this assessment.</span></div>}
    <Warnings items={response.warnings ?? []} />
    <details className="technical-details"><summary>View complete API response</summary><pre>{JSON.stringify(response, null, 2)}</pre></details>
  </div>
}

function InsightGroup({ icon, title, items, empty, wide = false }: {
  icon: ReactNode
  title: string
  items: Array<{ title: string; detail: string; evidenceFields: string[] }>
  empty: string
  wide?: boolean
}) {
  return <section className={`insight-group${wide ? ' insight-group-wide' : ''}`}>
    <div className="insight-group-title">{icon}<h4>{title}</h4></div>
    {items.length === 0 ? <p className="insight-empty">{empty}</p> : <div className="insight-items">{items.map((entry, index) => <article className="insight-item" key={`${entry.title}-${index}`}>
      <strong>{entry.title}</strong><p>{entry.detail}</p>
      <small>Response fields: {entry.evidenceFields.join(', ')}</small>
    </article>)}</div>}
  </section>
}
