import { CheckCircle2, CircleX } from 'lucide-react'

interface AttemptListProps {
  accepted?: Array<Record<string, unknown>>
  rejected?: Array<Record<string, unknown>>
  exerciseType?: string
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined) return 'Not available'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(2)
  if (typeof value === 'string' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value)
}

function acceptedMeasurement(item: Record<string, unknown>, exerciseType?: string): string {
  if (exerciseType === 'standing_broad_jump') {
    if (item.physical_distance_available === true && typeof item.distance_cm === 'number') {
      const meters = typeof item.distance_m === 'number' ? ` · ${item.distance_m.toFixed(3)} m` : ''
      return `${item.distance_cm.toFixed(1)} cm${meters}`
    }
    return `${displayValue(item.horizontal_displacement_px)} px · physical distance unavailable`
  }
  if (exerciseType === 'vertical_jump') {
    return typeof item.flight_time_height_cm === 'number'
      ? `${item.flight_time_height_cm.toFixed(2)} cm · flight-time estimate`
      : 'Height unavailable'
  }
  if (typeof item.duration_seconds === 'number') return `${item.duration_seconds.toFixed(2)} s`
  return 'Accepted'
}

function EventRows({ items, rejected = false, exerciseType }: { items: Array<Record<string, unknown>>; rejected?: boolean; exerciseType?: string }) {
  if (!items.length) return <div className="list-empty">No {rejected ? 'rejected' : 'accepted'} attempts reported.</div>
  return <div className="attempt-rows">{items.map((item, index) => (
    <article key={`${String(item.attempt_number ?? item.repetition_number ?? item.start_timestamp_ms ?? index)}-${index}`} className="attempt-row">
      <div className={`attempt-status ${rejected ? 'attempt-rejected' : ''}`}>{rejected ? <CircleX size={16} /> : <CheckCircle2 size={16} />}</div>
      <div className="attempt-main"><strong>{String(item.attempt_number ? `Attempt ${item.attempt_number}` : item.repetition_number ? `Repetition ${item.repetition_number}` : rejected ? `Rejected cycle ${index + 1}` : `Attempt ${index + 1}`)}</strong><span>{rejected ? displayValue(item.reason) : `${displayValue(item.start_timestamp_ms)} ms → ${displayValue(item.landing_timestamp_ms ?? item.end_timestamp_ms)} ms`}</span></div>
      <div className="attempt-value">{rejected ? 'Rejected' : acceptedMeasurement(item, exerciseType)}</div>
    </article>
  ))}</div>
}

export function AttemptList({ accepted = [], rejected = [], exerciseType }: AttemptListProps) {
  return <div className="attempt-list"><section><h3>Accepted <span>{accepted.length}</span></h3><EventRows items={accepted} exerciseType={exerciseType} /></section><section><h3>Rejected <span className="muted-count">{rejected.length}</span></h3><EventRows items={rejected} rejected exerciseType={exerciseType} /></section></div>
}
