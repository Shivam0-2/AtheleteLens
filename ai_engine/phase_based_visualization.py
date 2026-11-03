"""Render experimental phase-based push-up analysis over an input video.

Pose angles use the existing production analyzer's landmark-selection helper;
movement states and events come directly from the experimental phase detector.
The seven-frame rolling median matches the phase-analysis module and is
centered, so landmark measurements are collected before frames are rendered.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .phase_based_analysis import (
        DetectorConfig,
        MeasurementSeries,
        rolling_median,
        run_signal,
    )
    from .pushup_analysis import choose_elbow_angle
except ImportError:
    from phase_based_analysis import DetectorConfig, MeasurementSeries, rolling_median, run_signal
    from pushup_analysis import choose_elbow_angle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "models" / "pose_landmarker_lite.task"
DEFAULT_VIDEO = SCRIPT_DIR / "videos" / "pushup.mp4"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "experimental_outputs" / "phase_based"
MIN_VISIBILITY = 0.5  # Match the existing analyzer's default landmark selection.


@dataclass
class FramePose:
    """Small pose payload retained for the rendering pass."""

    points: list[tuple[float, float] | None]
    selected_side: str
    elbow_angle_deg: float | None
    tracking_visibility: float | None
    pose_detected: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an experimental phase-based push-up analysis video."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--output-name", default="pushup_phase_based_annotated.mp4"
    )
    parser.add_argument(
        "--summary-name", default="pushup_phase_based_visualization_summary.json"
    )
    return parser.parse_args()


def configured_detector() -> DetectorConfig:
    """Return the user-requested experimental detector configuration."""
    return DetectorConfig(
        bottom_angle_deg=100.0,
        top_angle_deg=140.0,
        hysteresis_deg=5.0,
        confirmation_frames=2,
        min_repetition_duration_s=0.25,
        max_repetition_duration_s=3.0,
        max_measurement_gap_s=0.25,
    )


def _pose_payload(result: Any) -> FramePose:
    """Extract the first pose and use the analyzer's side/angle selection."""
    if not result.pose_landmarks:
        return FramePose([], "", None, None, False)

    pose = result.pose_landmarks[0]
    points: list[tuple[float, float] | None] = []
    for landmark in pose:
        try:
            x, y = float(landmark.x), float(landmark.y)
            points.append((x, y) if math.isfinite(x) and math.isfinite(y) else None)
        except (AttributeError, TypeError, ValueError):
            points.append(None)

    angle, side, _left_angle, _right_angle, fields = choose_elbow_angle(
        pose, MIN_VISIBILITY
    )
    if angle is not None and not math.isfinite(angle):
        angle = None

    visibility_values = [
        fields.get(f"{side}_{joint}_visibility")
        for joint in ("shoulder", "elbow", "wrist")
    ] if side else []
    supplied_visibility = [
        float(value)
        for value in visibility_values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    tracking_visibility = (
        min(supplied_visibility) if len(supplied_visibility) == 3 else None
    )
    return FramePose(points, side, angle, tracking_visibility, True)


def _draw_pose(
    frame: Any,
    points: list[tuple[float, float] | None],
    connections: list[Any],
    cv2: Any,
) -> None:
    """Draw the MediaPipe pose connections and available landmark points."""
    height, width = frame.shape[:2]

    def pixel(point: tuple[float, float]) -> tuple[int, int]:
        x = min(width - 1, max(0, int(round(point[0] * (width - 1)))) )
        y = min(height - 1, max(0, int(round(point[1] * (height - 1)))) )
        return x, y

    for connection in connections:
        start_index, end_index = int(connection.start), int(connection.end)
        if start_index >= len(points) or end_index >= len(points):
            continue
        start, end = points[start_index], points[end_index]
        if start is not None and end is not None:
            cv2.line(frame, pixel(start), pixel(end), (45, 220, 70), 2, cv2.LINE_AA)

    for point in points:
        if point is not None:
            cv2.circle(frame, pixel(point), 3, (20, 235, 250), -1, cv2.LINE_AA)


def _draw_overlay(
    frame: Any,
    phase: str,
    repetition_count: int,
    angle: float | None,
    side: str,
    pose_detected: bool,
    visibility: float | None,
    accepted_marker: int | None,
    marker_frames_left: int,
    cv2: Any,
) -> None:
    """Draw phase, count, angle, tracking quality, and transient rep marker."""
    height, width = frame.shape[:2]
    panel_width = min(440, max(300, width - 20))
    panel_height = 156
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (20, 24, 31), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)

    angle_text = f"{angle:.1f} deg ({side})" if angle is not None else "unavailable"
    if not pose_detected:
        tracking_text, tracking_color = "NO POSE", (70, 80, 240)
    elif angle is None:
        tracking_text, tracking_color = "POSE FOUND / ANGLE INVALID", (30, 175, 255)
    elif visibility is None:
        tracking_text, tracking_color = "POSE TRACKED / VISIBILITY N/A", (30, 205, 235)
    elif visibility >= 0.8:
        tracking_text, tracking_color = f"GOOD ({visibility:.2f})", (70, 220, 100)
    elif visibility >= MIN_VISIBILITY:
        tracking_text, tracking_color = f"FAIR ({visibility:.2f})", (30, 190, 245)
    else:
        tracking_text, tracking_color = f"LOW ({visibility:.2f})", (70, 80, 240)

    lines = [
        ("AthleteLens | EXPERIMENTAL", (245, 245, 245)),
        (f"Phase: {phase}", (255, 215, 100)),
        (f"Accepted repetitions: {repetition_count}", (245, 245, 245)),
        (f"Elbow angle (median 7): {angle_text}", (245, 245, 245)),
        (f"Tracking: {tracking_text}", tracking_color),
    ]
    for index, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (22, 35 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            color,
            1 if index == 0 else 2,
            cv2.LINE_AA,
        )

    if accepted_marker is not None and marker_frames_left > 0:
        text = f"REP {accepted_marker} ACCEPTED"
        size, _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.05, 3)
        x = max(10, (width - size[0]) // 2)
        y = max(panel_height + 50, height // 5)
        cv2.rectangle(frame, (x - 16, y - size[1] - 16), (x + size[0] + 16, y + 14), (20, 145, 35), -1)
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_DUPLEX,
            1.05,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "A required dependency is unavailable. Activate ai_engine/.venv "
            "and install ai_engine/requirements.txt."
        ) from exc
    return cv2, mp, np


def render_video(
    video_path: Path, model_path: Path, output_dir: Path, output_name: str, summary_name: str
) -> dict[str, Any]:
    """Detect once, reuse detector output, render every frame, and write summary."""
    cv2, mp, np = _load_dependencies()
    config = configured_detector()
    config.validate()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open input video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(
            f"Invalid input video metadata (FPS={fps}, dimensions={width}x{height})"
        )

    timestamps: list[int] = []
    frame_numbers: list[int] = []
    angles: list[float | None] = []
    poses: list[FramePose] = []
    previous_timestamp_ms = -1
    landmarker_context = None
    try:
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
        try:
            landmarker_context = mp.tasks.vision.PoseLandmarker.create_from_options(
                options
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Could not initialize Pose Landmarker from '{model_path}'. "
                "Check that it is a compatible .task model."
            ) from exc

        with landmarker_context as landmarker:
            frame_index = 0
            while True:
                success, frame_bgr = capture.read()
                if not success:
                    break
                timestamp_ms = max(
                    previous_timestamp_ms + 1,
                    int(round(frame_index * 1000.0 / fps)),
                )
                previous_timestamp_ms = timestamp_ms
                rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
                )
                payload = _pose_payload(result)
                angle = payload.elbow_angle_deg
                if angle is not None and not math.isfinite(angle):
                    angle = None
                frame_numbers.append(frame_index + 1)
                timestamps.append(timestamp_ms)
                angles.append(angle)
                poses.append(payload)
                frame_index += 1
    finally:
        capture.release()

    if not angles:
        raise ValueError(f"Input video contains no readable frames: {video_path}")

    series = MeasurementSeries(
        label=video_path.stem,
        source_path=video_path.resolve(),
        frame_numbers=frame_numbers,
        timestamps_ms=timestamps,
        angles=angles,
    )
    smoothed = rolling_median(angles, 7)
    state_rows, event_rows, repetition_count = run_signal(
        series, "rolling_median", smoothed, config
    )
    if len(state_rows) != len(poses):
        raise RuntimeError("Detector output frame count does not match video frame count")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_dir / output_name).resolve()
    summary_path = (output_dir / summary_name).resolve()
    if output_path == video_path.resolve():
        raise ValueError("Output path must not overwrite the input video")
    if summary_path in {video_path.resolve(), output_path}:
        raise ValueError("Summary path must be distinct from input and output video paths")

    pose_connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
    writer = None
    output_capture = cv2.VideoCapture(str(video_path))
    if not output_capture.isOpened():
        output_capture.release()
        raise ValueError(f"Could not reopen input video for rendering: {video_path}")

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create MP4 output: {output_path}")

        accepted_events = [event for event in event_rows if event["event_type"] == "completed"]
        accepted_by_number = {
            int(event["repetition_number"]): event for event in accepted_events
        }
        previous_count = 0
        marker_rep: int | None = None
        marker_frames_left = 0
        marker_duration_frames = max(1, int(round(fps * 0.7)))

        for index, row in enumerate(state_rows):
            success, frame_bgr = output_capture.read()
            if not success:
                raise ValueError(
                    f"Input video ended during rendering at frame {index + 1}"
                )
            current_count = int(row["repetition_count"])
            if current_count > previous_count:
                marker_rep = current_count
                marker_frames_left = marker_duration_frames
            _draw_pose(frame_bgr, poses[index].points, pose_connections, cv2)
            _draw_overlay(
                frame_bgr,
                row["movement_state"],
                current_count,
                smoothed[index],
                poses[index].selected_side,
                poses[index].pose_detected,
                poses[index].tracking_visibility,
                marker_rep,
                marker_frames_left,
                cv2,
            )
            writer.write(frame_bgr)
            if marker_frames_left > 0:
                marker_frames_left -= 1
            previous_count = current_count

        # Ensure output had exactly one rendered frame for each input frame.
        extra_success, _extra_frame = output_capture.read()
        if extra_success:
            raise RuntimeError("Input frame count changed between detection and rendering passes")
    finally:
        output_capture.release()
        if writer is not None:
            writer.release()

    confirmation_frames: dict[int, tuple[int, int]] = {}
    prior_count = 0
    for row in state_rows:
        current_count = int(row["repetition_count"])
        if current_count > prior_count:
            confirmation_frames[current_count] = (
                int(row["frame_number"]), int(row["timestamp_ms"])
            )
        prior_count = current_count

    accepted_summary = []
    for event in accepted_events:
        repetition_number = int(event["repetition_number"])
        confirmed_at = confirmation_frames.get(repetition_number)
        accepted_summary.append(
            {
                "repetition_number": repetition_number,
                "cycle_start_timestamp_ms": int(event["start_timestamp_ms"]),
                "detector_event_timestamp_ms": int(event["end_timestamp_ms"]),
                "accepted_on_frame": confirmed_at[0] if confirmed_at else None,
                "accepted_at_timestamp_ms": confirmed_at[1] if confirmed_at else None,
                "duration_seconds": float(event["duration_seconds"]),
            }
        )

    summary = {
        "status": "Experimental visualization; not a validated sports assessment.",
        "input_video": str(video_path.resolve()),
        "output_video": str(output_path),
        "frames_processed": len(poses),
        "fps": fps,
        "dimensions": {"width": width, "height": height},
        "signal": "centered rolling median of 7 valid contiguous frames; missing values left blank",
        "landmark_selection": {
            "method": "existing choose_elbow_angle helper",
            "minimum_visibility": MIN_VISIBILITY,
        },
        "configuration": asdict(config),
        "detected_repetitions": repetition_count,
        "accepted_repetition_timestamps": accepted_summary,
        "rejected_events": [
            {
                "reason": event["reason"],
                "start_timestamp_ms": event["start_timestamp_ms"],
                "end_timestamp_ms": event["end_timestamp_ms"],
                "duration_seconds": event["duration_seconds"],
            }
            for event in event_rows
            if event["event_type"] == "rejected"
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not video_path.is_file():
        print(f"Error: video file does not exist: {video_path}", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"Error: model file does not exist: {model_path}", file=sys.stderr)
        return 2
    try:
        summary = render_video(
            video_path,
            model_path,
            args.output_dir.expanduser().resolve(),
            args.output_name,
            args.summary_name,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: rendering failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2

    print(f"Annotated video: {summary['output_video']}")
    print(f"Frames: {summary['frames_processed']}; repetitions: {summary['detected_repetitions']}")
    print(f"Summary: {args.output_dir.expanduser().resolve() / args.summary_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
