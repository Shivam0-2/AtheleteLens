import { Activity, CheckCircle2, ScanLine } from 'lucide-react'
import type { AssessmentResponse } from '../types'

export function QualitySummary({ response }: { response?: AssessmentResponse }) {
  if (!response) return <div className="empty-panel"><ScanLine size={19} /><span>Quality indicators will appear after a video assessment is submitted.</span></div>
  const coverage = response.measurement_coverage_percentage
  return (
    <div className="quality-grid">
      <div className="quality-card"><div className="quality-icon"><Activity size={17} /></div><span>Pose detection</span><strong>{response.pose_detection_percentage.toFixed(1)}<small>%</small></strong><small>{response.frames_with_detected_poses} / {response.total_frames_processed} frames</small></div>
      <div className="quality-card"><div className="quality-icon"><CheckCircle2 size={17} /></div><span>Valid measurements</span><strong>{coverage === undefined ? '—' : `${coverage.toFixed(1)}%`}</strong><small>{response.frames_with_valid_measurements ?? response.frames_with_valid_elbow_angle ?? 'Not reported'} valid frames</small></div>
    </div>
  )
}
