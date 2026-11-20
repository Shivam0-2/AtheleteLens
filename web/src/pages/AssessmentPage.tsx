import { useEffect, useRef, useState, type FormEvent, type MouseEvent } from 'react'
import { ArrowLeft, ArrowRight, Camera, CircleAlert, LoaderCircle, Ruler, Video } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { submitAssessment, AssessmentApiError } from '../api/client'
import { useDemoAuth } from '../auth/AuthContext'
import { VideoPicker } from '../components/VideoPicker'
import { getExercise } from '../data/exercises'
import { saveAssessment } from '../data/assessmentStore'

type UploadState = 'idle' | 'selected' | 'uploading' | 'success' | 'failed'
interface ImagePoint { x: number; y: number }

const supportedSuffixes = new Set(['.avi', '.m4v', '.mkv', '.mov', '.mp4', '.webm'])

function supportedVideo(file: File): boolean {
  const suffix = file.name.toLowerCase().slice(file.name.lastIndexOf('.'))
  return supportedSuffixes.has(suffix)
}

export function AssessmentPage() {
  const { exercise: exerciseKey } = useParams()
  const exercise = getExercise(exerciseKey)
  const navigate = useNavigate()
  const { session } = useDemoAuth()
  const videoRef = useRef<HTMLVideoElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 })
  const [points, setPoints] = useState<ImagePoint[]>([])
  const [knownLength, setKnownLength] = useState('')
  const [extendedAngle, setExtendedAngle] = useState('')
  const [flexedAngle, setFlexedAngle] = useState('')
  const [uploadState, setUploadState] = useState<UploadState>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      setNaturalSize({ width: 0, height: 0 })
      setPoints([])
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  if (!exercise) return <section className="empty-state"><CircleAlert size={25} /><h2>Assessment not found</h2><p>Choose one of the available movement protocols.</p><Link to="/exercises" className="button button-secondary">Back to assessments</Link></section>
  const selectedExercise = exercise

  function selectFile(next: File | null) {
    setErrorMessage(null)
    setPoints([])
    setNaturalSize({ width: 0, height: 0 })
    setFile(next)
    setUploadState(next ? 'selected' : 'idle')
    if (next && !supportedVideo(next)) {
      setFile(null)
      setUploadState('failed')
      setErrorMessage('Choose an MP4, MOV, AVI, MKV, M4V, or WebM video file.')
    }
  }

  function selectCalibrationPoint(event: MouseEvent<HTMLVideoElement>) {
    if (!naturalSize.width || !naturalSize.height || points.length >= 2) return
    const video = event.currentTarget
    const rect = event.currentTarget.getBoundingClientRect()
    const relativeX = event.clientX - rect.left
    const relativeY = event.clientY - rect.top
    // Ignore the native control strip: its clicks are not points on the video image.
    if (relativeY > rect.height - Math.min(46, rect.height * 0.13)) return
    const scale = Math.min(rect.width / naturalSize.width, rect.height / naturalSize.height)
    const displayedWidth = naturalSize.width * scale
    const displayedHeight = naturalSize.height * scale
    const offsetX = (rect.width - displayedWidth) / 2
    const offsetY = (rect.height - displayedHeight) / 2
    const imageX = (relativeX - offsetX) / scale
    const imageY = (relativeY - offsetY) / scale
    if (imageX < 0 || imageX >= naturalSize.width || imageY < 0 || imageY >= naturalSize.height) return
    video.pause()
    const x = Math.round(imageX)
    const y = Math.round(imageY)
    setPoints((current) => current.length >= 2 ? [ { x, y } ] : [...current, { x, y }])
    setErrorMessage(null)
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (uploadState === 'uploading') return
    if (!file) {
      setUploadState('failed')
      setErrorMessage('Select a video before submitting the assessment.')
      return
    }

    let config: Record<string, unknown> | undefined
    if (selectedExercise.key === 'curlups') {
      const extended = Number(extendedAngle)
      const flexed = Number(flexedAngle)
      if (!extendedAngle || !flexedAngle || !Number.isFinite(extended) || !Number.isFinite(flexed)
          || flexed < 0 || extended > 180 || flexed >= extended) {
        setUploadState('failed')
        setErrorMessage('Enter angle thresholds with 0 ≤ flexed angle < extended angle ≤ 180 degrees.')
        return
      }
      config = { extended_angle_deg: extended, flexed_angle_deg: flexed }
    }

    if (selectedExercise.key === 'broad-jumps' && (points.length > 0 || knownLength.trim())) {
      const length = Number(knownLength)
      if (points.length !== 2) {
        setUploadState('failed')
        setErrorMessage('Select both endpoints of the ground reference, or clear the calibration inputs to submit in pixels only.')
        return
      }
      if (!Number.isFinite(length) || length <= 0) {
        setUploadState('failed')
        setErrorMessage('Enter a positive, finite reference length in meters.')
        return
      }
      const dx = points[1].x - points[0].x
      const dy = points[1].y - points[0].y
      const referencePixels = Math.hypot(dx, dy)
      if (referencePixels < 2) {
        setUploadState('failed')
        setErrorMessage('The reference points are too close together. Select two clearly separated points.')
        return
      }
      if (Math.atan2(Math.abs(dy), Math.abs(dx)) * 180 / Math.PI > 15) {
        setUploadState('failed')
        setErrorMessage('The reference should align with the horizontal jump direction in the video (within 15°).')
        return
      }
      const approximateScale = referencePixels / length
      if (approximateScale < 10 || approximateScale > 100_000) {
        setUploadState('failed')
        setErrorMessage('This reference length and point spacing produce a scale outside the API calibration limits.')
        return
      }
      config = {
        reference_endpoint_a_x_px: points[0].x,
        reference_endpoint_a_y_px: points[0].y,
        reference_endpoint_b_x_px: points[1].x,
        reference_endpoint_b_y_px: points[1].y,
        reference_length_m: length,
      }
    }

    setUploadState('uploading')
    setErrorMessage(null)
    try {
      const submission = await submitAssessment(selectedExercise.key, file, config, session?.email ?? 'demo-athlete')
      const response = submission.response
      const saved = saveAssessment(response, session?.role === 'admin' ? 'admin' : 'athlete')
      setUploadState('success')
      const recordId = submission.assessmentId ?? saved?.id
      navigate(recordId ? `/results/${recordId}` : '/results', {
        state: { response, exerciseKey: selectedExercise.key, recordId, persistenceUnavailable: !submission.assessmentId },
      })
    } catch (error) {
      setUploadState('failed')
      setErrorMessage(error instanceof AssessmentApiError ? error.message : 'The assessment could not be completed. Please try again.')
    }
  }

  const broad = selectedExercise.key === 'broad-jumps'
  const curlup = selectedExercise.key === 'curlups'
  const isUploading = uploadState === 'uploading'

  return <div className="page-stack">
    <Link to="/exercises" className="back-link"><ArrowLeft size={15} /> Assessment library</Link>
    <section className="page-intro assessment-intro"><span className="eyebrow">{selectedExercise.focus}</span><h2>{selectedExercise.name}</h2><p>{selectedExercise.description}</p></section>
    <form className="assessment-layout" onSubmit={handleSubmit}>
      <div className="form-panel">
        <div className="panel-heading"><div><span className="eyebrow">STEP 01</span><h3>Choose a video</h3></div><Video size={19} /></div>
        <VideoPicker onFileSelected={selectFile} />
        <div className="setup-tips"><div className="tip-heading"><Camera size={16} /> Recording checklist</div><p>Keep the athlete and relevant joints visible. Use a stable camera and good lighting, and record the complete movement.</p></div>

        {curlup && <div className="config-placeholder"><span className="eyebrow">REQUIRED CONFIGURATION</span><p>The curl-up API requires included-angle thresholds. No standardized defaults are defined for this experimental protocol.</p><div className="field-row"><label>Extended angle (°)<input type="number" min="0" max="180" value={extendedAngle} onChange={(event) => setExtendedAngle(event.target.value)} placeholder="e.g. 115" required /></label><label>Flexed angle (°)<input type="number" min="0" max="180" value={flexedAngle} onChange={(event) => setFlexedAngle(event.target.value)} placeholder="e.g. 100" required /></label></div></div>}

        {broad && <div className="config-placeholder"><span className="eyebrow">OPTIONAL DISTANCE CALIBRATION</span><p>Place a known-length reference flat on the ground, parallel to the jump direction and close to the landing path. Click its two endpoints in the video.</p>
          {previewUrl ? <div className="calibration-video-panel">
            <div className="calibration-frame" style={naturalSize.width && naturalSize.height ? { maxWidth: `${Math.min(920, 420 * naturalSize.width / naturalSize.height)}px` } : undefined}>
              <video ref={videoRef} src={previewUrl} controls preload="metadata" onLoadedMetadata={(event) => setNaturalSize({ width: event.currentTarget.videoWidth, height: event.currentTarget.videoHeight })} onClick={selectCalibrationPoint} />
              {naturalSize.width > 0 && <svg className="calibration-overlay" viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`} preserveAspectRatio="none" aria-hidden="true">
                {points.length === 2 && <line x1={points[0].x} y1={points[0].y} x2={points[1].x} y2={points[1].y} />}
                {points.map((point, index) => <g key={`${point.x}-${point.y}`}><circle cx={point.x} cy={point.y} r={Math.max(5, naturalSize.width / 100)} /><text x={point.x + 10} y={point.y - 10}> {index === 0 ? 'A' : 'B'} </text></g>)}
              </svg>}
            </div>
            <div className="calibration-controls"><span>{naturalSize.width ? `Source frame ${naturalSize.width} × ${naturalSize.height}` : 'Loading video dimensions…'}</span><button type="button" className="text-button" onClick={() => setPoints([])}>Clear points</button></div>
            <p className="point-count">{points.length === 0 ? 'Select point A, then point B.' : points.length === 1 ? `A selected at (${points[0].x}, ${points[0].y}) px · select point B.` : `A (${points[0].x}, ${points[0].y}) px · B (${points[1].x}, ${points[1].y}) px`}</p>
          </div> : <div className="calibration-preview">Choose a video to mark the reference endpoints.</div>}
          <label className="field-label"><Ruler size={15} /> Known distance between points (m)<input type="number" min="0.001" step="0.01" value={knownLength} onChange={(event) => setKnownLength(event.target.value)} placeholder="e.g. 2.00" /></label>
          <small className="muted-copy">Calibration is local to the reference ground plane and does not correct perspective distortion. If omitted, the API returns pixel displacement only.</small>
        </div>}

        {errorMessage && <div className="form-error" role="alert"><CircleAlert size={16} /><span>{errorMessage}</span></div>}
        {uploadState === 'failed' && <button type="button" className="text-button retry-button" onClick={() => { setErrorMessage(null); setUploadState(file ? 'selected' : 'idle') }}>Dismiss error</button>}
        <button className="button button-primary submit-placeholder" type="submit" disabled={isUploading || uploadState === 'success' || !file}>
          <span>{isUploading ? 'Uploading and processing…' : uploadState === 'success' ? 'Assessment complete' : file ? 'Submit assessment' : 'Choose a video to continue'}</span>
          {isUploading ? <LoaderCircle className="spin-icon" size={16} /> : <ArrowRight size={16} />}
        </button>
        {isUploading && <p className="processing-note">The video is being analyzed by the pose pipeline. Keep this page open until processing finishes.</p>}
      </div>
      <aside className="assessment-side"><div className="side-panel"><div className="side-panel-icon">{selectedExercise.icon}</div><span className="eyebrow">PROTOCOL OVERVIEW</span><h3>{selectedExercise.shortName}</h3><p>{selectedExercise.description}</p><div className="protocol-row"><span>Endpoint</span><code>POST {selectedExercise.endpoint}</code></div><div className="protocol-row"><span>Configuration</span><span>{curlup ? 'Required thresholds' : broad ? 'Optional calibration' : 'Server defaults'}</span></div></div><div className="notice-card"><CircleAlert size={17} /><p>Assessment outputs are experimental and have not been scientifically validated.</p></div></aside>
    </form>
  </div>
}
