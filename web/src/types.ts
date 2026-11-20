export type ExerciseKey = 'pushups' | 'curlups' | 'vertical-jumps' | 'broad-jumps'

export interface ExerciseInfo {
  key: ExerciseKey
  name: string
  shortName: string
  description: string
  focus: string
  icon: string
  endpoint: string
}

export interface BaseAssessmentResponse {
  status?: string
  processing_status: string
  assessment_notice?: string
  input_filename: string
  exercise_type: string
  video: { fps: number; width: number; height: number }
  total_frames_processed: number
  frames_with_detected_poses: number
  frames_with_valid_measurements?: number
  frames_with_valid_elbow_angle?: number
  pose_detection_percentage: number
  measurement_coverage_percentage?: number
  measurement_quality?: Record<string, unknown>
  detected_repetitions?: number
  average_elbow_angle_degrees?: number | null
  minimum_elbow_angle_degrees?: number | null
  maximum_elbow_angle_degrees?: number | null
  accepted_repetition_timestamps?: Array<Record<string, unknown>>
  rejected_events?: Array<Record<string, unknown>>
  accepted_attempt_count?: number
  accepted_attempts?: Array<Record<string, unknown>>
  rejected_attempt_count?: number
  rejected_attempts?: Array<Record<string, unknown>>
  measurement_summary?: Record<string, unknown>
  physical_distance_attempt_count?: number | null
  physical_distance_available?: boolean | null
  detector_configuration?: Record<string, unknown>
  annotated_video_available?: boolean
  annotated_video_url?: string | null
  annotated_video_path?: string | null
  warnings?: string[]
}

export interface PushUpAssessmentResponse extends BaseAssessmentResponse {
  exercise_type: 'push_up'
  detected_repetitions: number
  frames_with_valid_elbow_angle: number
  average_elbow_angle_degrees: number | null
  minimum_elbow_angle_degrees: number | null
  maximum_elbow_angle_degrees: number | null
  accepted_repetition_timestamps: Array<Record<string, unknown>>
  rejected_events: Array<Record<string, unknown>>
}

export interface ProtocolAssessmentResponse extends BaseAssessmentResponse {
  exercise_type: 'curl_up' | 'vertical_jump' | 'standing_broad_jump'
  frames_with_valid_measurements: number
  measurement_coverage_percentage: number
  measurement_quality: Record<string, unknown>
  accepted_attempt_count: number
  accepted_attempts: Array<Record<string, unknown>>
  rejected_attempt_count: number
  rejected_attempts: Array<Record<string, unknown>>
  measurement_summary: Record<string, unknown>
  physical_distance_attempt_count: number | null
  physical_distance_available: boolean | null
  detector_configuration: Record<string, unknown>
  annotated_video_available: boolean
  annotated_video_path: string | null
}

export type AssessmentResponse = PushUpAssessmentResponse | ProtocolAssessmentResponse

export interface AssessmentRecord {
  assessment_id: string
  athlete_id: string
  exercise_type: string
  submitted_at: string
  response: AssessmentResponse
  configuration: Record<string, unknown>
  algorithm_version: string | null
  annotated_video_id: string | null
  annotated_video_url: string | null
}

export interface RequestError {
  message: string
  status?: number
  detail?: unknown
}
