import { ArrowUpRight } from 'lucide-react'
import { ExerciseCard } from '../components/ExerciseCard'
import { exercises } from '../data/exercises'

export function ExercisesPage() {
  return <div className="page-stack"><section className="page-intro"><span className="eyebrow">FOUR MOVEMENT PROTOCOLS</span><h2>Assessment library</h2><p>Select a protocol to prepare a video and review its configuration. Results come directly from the AthleteLens API.</p></section><div className="exercise-grid exercise-grid-large">{exercises.map((exercise, index) => <ExerciseCard key={exercise.key} exercise={exercise} index={index} />)}</div><div className="inline-note"><ArrowUpRight size={16} /><span>Each protocol has different measurement assumptions. Review the setup notes before recording.</span></div></div>
}
