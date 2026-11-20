import { useEffect, useState } from 'react'
import { ArrowRight, Beaker, CircleAlert, CircleHelp, Clock3, Dumbbell, FileClock, ScanLine, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import { ExerciseCard } from '../components/ExerciseCard'
import { exercises } from '../data/exercises'
import { useDemoAuth } from '../auth/AuthContext'
import { getAthleteAssessmentHistory } from '../api/client'
import type { AssessmentRecord } from '../types'

function exerciseLabel(record: AssessmentRecord): string {
  const type = record.exercise_type
  const routeKey = Object.entries({
    pushups: 'push_up', curlups: 'curl_up', 'vertical-jumps': 'vertical_jump', 'broad-jumps': 'standing_broad_jump',
  }).find(([, exerciseType]) => exerciseType === type)?.[0]
  return exercises.find((exercise) => exercise.key === routeKey)?.name ?? type.replaceAll('_', ' ')
}

function acceptedLabel(record: AssessmentRecord): string {
  const response = record.response
  const count = response.exercise_type === 'push_up' ? response.detected_repetitions : response.accepted_attempt_count
  return `${count} ${response.exercise_type === 'push_up' || response.exercise_type === 'curl_up' ? 'repetitions' : 'attempts'} reported`
}

export function DashboardPage() {
  const { session } = useDemoAuth()
  const [history, setHistory] = useState<AssessmentRecord[]>([])
  const [historyError, setHistoryError] = useState<string | null>(null)
  useEffect(() => {
    if (!session) return
    getAthleteAssessmentHistory(session.email).then(setHistory).catch((error: unknown) => {
      setHistoryError(error instanceof Error ? error.message : 'Assessment history is unavailable.')
    })
  }, [session])
  return <div className="page-stack">
    <section className="welcome-panel"><div className="welcome-copy"><span className="eyebrow blue-eyebrow"><Sparkles size={13} /> MOVEMENT, MADE VISIBLE</span><h2>Train with a clearer<br /><em>view of your movement.</em></h2><p>Explore preliminary video-based assessments designed to help you review movement patterns and performance signals.</p><Link to="/exercises" className="button button-primary">Start an assessment <ArrowRight size={16} /></Link></div><div className="welcome-art" aria-hidden="true"><div className="art-ring ring-one" /><div className="art-ring ring-two" /><div className="art-core"><ScanLine size={37} /></div><span className="art-label label-top">POSE TRACKING</span><span className="art-label label-bottom">4 PROTOCOLS</span></div></section>
    <div className="section-heading"><div><span className="eyebrow">ASSESSMENT BATTERY</span><h2>Choose your movement</h2></div><Link to="/exercises" className="text-link">View all <ArrowRight size={15} /></Link></div>
    <div className="exercise-grid">{exercises.map((exercise, index) => <ExerciseCard key={exercise.key} exercise={exercise} index={index} />)}</div>
    <section className="history-panel"><div className="history-heading"><div><span className="eyebrow">SAVED ON THIS ATHLETELENS SERVER</span><h2>Assessment history</h2><p>{session?.displayName ?? 'Athlete demo'} · demo identity association</p></div><FileClock size={20} /></div>{history.length ? <div className="history-list">{history.map((item) => <Link key={item.assessment_id} to={`/results/${item.assessment_id}`} className="history-row"><div className="history-exercise-mark">{exerciseLabel(item).slice(0, 1).toUpperCase()}</div><div className="history-main"><strong>{exerciseLabel(item)}</strong><span>{item.response.input_filename} · {new Date(item.submitted_at).toLocaleString()}</span></div><div className="history-result">{acceptedLabel(item)}</div><ArrowRight size={15} /></Link>)}</div> : <div className="history-empty">{historyError ? <><CircleAlert size={16} />{historyError}</> : 'No persisted assessments are associated with this demo identity yet.'}</div>}</section>
    <section className="dashboard-bottom"><div className="info-card"><div className="info-icon"><Beaker size={18} /></div><div><strong>Research prototype</strong><p>Assessments are experimental and are not scientifically validated or a substitute for professional evaluation.</p></div></div><div className="info-card"><div className="info-icon"><CircleHelp size={18} /></div><div><strong>How it works</strong><p>Upload a clear, full-body video. The analysis reports only measurements returned by the assessment service.</p></div></div></section>
    <div className="dashboard-caption"><span><Clock3 size={14} /> {history.length} persisted response{history.length === 1 ? '' : 's'} associated with this demo identity.</span><span><Dumbbell size={14} /> AthleteLens performance lab</span></div>
  </div>
}
