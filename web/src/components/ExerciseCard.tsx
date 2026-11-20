import { ArrowUpRight, Check } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { ExerciseInfo } from '../types'

export function ExerciseCard({ exercise, index = 0 }: { exercise: ExerciseInfo; index?: number }) {
  return (
    <Link to={`/assessments/${exercise.key}`} className={`exercise-card accent-${index}`}>
      <div className="exercise-card-top"><div className="exercise-symbol">{exercise.icon}</div><span className="card-arrow"><ArrowUpRight size={17} /></span></div>
      <div className="exercise-card-copy"><span className="card-kicker">PROTOCOL 0{index + 1}</span><h3>{exercise.name}</h3><p>{exercise.description}</p></div>
      <div className="exercise-card-bottom"><span>{exercise.focus}</span><span className="check-mark"><Check size={13} /></span></div>
    </Link>
  )
}
