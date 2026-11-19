"""Run an unvalidated curl-up diagnostic and export video, CSV, JSON, and plot.

Pose inference and timestamps come from ``PoseVideoPipeline``. This command
does not tune thresholds: the caller must provide all repetition settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from athletelens_ai.assessments.curl_up import CurlUpAssessment, CurlUpConfig
from athletelens_ai.pipeline import PoseVideoPipeline
from athletelens_ai.video import iter_video_frames, open_video
from athletelens_ai.visualization import draw_normalized_points


AI_ENGINE_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = AI_ENGINE_DIR / "videos" / "curl up.mp4"
DEFAULT_MODEL = AI_ENGINE_DIR / "models" / "pose_landmarker_lite.task"
DEFAULT_OUTPUT_DIR = AI_ENGINE_DIR / "experimental_outputs" / "curl_up_diagnostic"

LANDMARK_VISIBILITY_COLUMNS = {
    "left_shoulder_visibility": 11,
    "left_hip_visibility": 23,
    "left_knee_visibility": 25,
    "right_shoulder_visibility": 12,
    "right_hip_visibility": 24,
    "right_knee_visibility": 26,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing experimental curl-up assessment on a video. "
            "Thresholds are required and are never tuned by this command."
        )
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extended-angle-deg", required=True, type=float)
    parser.add_argument("--flexed-angle-deg", required=True, type=float)
    parser.add_argument("--hysteresis-deg", required=True, type=float)
    parser.add_argument("--confirmation-frames", required=True, type=int)
    parser.add_argument("--min-repetition-duration-s", required=True, type=float)
    parser.add_argument("--max-repetition-duration-s", required=True, type=float)
    parser.add_argument("--max-measurement-gap-s", required=True, type=float)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--selected-side", choices=("auto", "left", "right"), default="auto")
    parser.add_argument(
        "--test-duration-s",
        type=float,
        default=None,
        help="Optional analysis window from video start; omit to analyze the full clip.",
    )
    parser.add_argument(
        "--manual-reference-repetitions",
        type=int,
        default=None,
        help="Optional approximate manual count to include as a comparison reference.",
    )
    return parser.parse_args()


def _make_config(args: argparse.Namespace, aspect_ratio: float) -> CurlUpConfig:
    config = CurlUpConfig(
        image_aspect_ratio=aspect_ratio,
        extended_angle_deg=args.extended_angle_deg,
        flexed_angle_deg=args.flexed_angle_deg,
        hysteresis_deg=args.hysteresis_deg,
        confirmation_frames=args.confirmation_frames,
        min_repetition_duration_s=args.min_repetition_duration_s,
        max_repetition_duration_s=args.max_repetition_duration_s,
        max_measurement_gap_s=args.max_measurement_gap_s,
        min_visibility=args.min_visibility,
        starting_position="extended",
        selected_side=args.selected_side,
        test_duration_s=args.test_duration_s,
    )
    config.validate()
    return config


def _finite_point(landmark: dict[str, Any]) -> tuple[float, float] | None:
    try:
        x = float(landmark["x"])
        y = float(landmark["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _pose_points(observation: Any) -> list[tuple[float, float] | None]:
    if not observation.pose_landmark_data:
        return []
    return [_finite_point(item) for item in observation.pose_landmark_data[0]]


def _landmark_visibility(observation: Any, field: str) -> float | None:
    if not observation.pose_landmark_data:
        return None
    index = LANDMARK_VISIBILITY_COLUMNS[field]
    if index >= len(observation.pose_landmark_data[0]):
        return None
    value = observation.pose_landmark_data[0][index].get("visibility")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _write_measurements_csv(
    path: Path,
    rows: list[dict[str, Any]],
    observations: list[Any],
    accepted_timestamps: set[int],
) -> None:
    columns = [
        "frame_number",
        "timestamp_ms",
        "pose_detected",
        "measurement_valid",
        "selected_side",
        "included_angle_deg",
        "curl_flexion_deg",
        "visibility_score",
        *LANDMARK_VISIBILITY_COLUMNS,
        "missing_landmarks",
        "movement_state",
        "repetition_count",
        "accepted_repetition",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row, observation in zip(rows, observations):
            values = dict(row)
            values["pose_detected"] = bool(observation.pose_detected)
            values["accepted_repetition"] = row["timestamp_ms"] in accepted_timestamps
            for field in LANDMARK_VISIBILITY_COLUMNS:
                values[field] = _landmark_visibility(observation, field)
            writer.writerow({key: values.get(key) for key in columns})


def _draw_overlay(
    frame: Any,
    row: dict[str, Any],
    visibility: float | None,
    accepted_number: int | None,
    cv2: Any,
) -> None:
    height, width = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (8, 8), (min(width - 8, 465), 128), (25, 28, 35), -1)
    cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)
    angle = row["included_angle_deg"]
    angle_text = f"{float(angle):.1f} deg" if angle is not None else "unavailable"
    if not row["pose_detected"]:
        quality, color = "NO POSE", (60, 70, 240)
    elif not row["measurement_valid"]:
        quality, color = "LANDMARKS / ANGLE INVALID", (20, 165, 250)
    else:
        score = "n/a" if visibility is None else f"{visibility:.2f}"
        quality, color = f"TRACKED (visibility {score})", (65, 215, 105)
    lines = [
        ("AthleteLens | EXPERIMENTAL / UNVALIDATED", (245, 245, 245)),
        (f"State: {row['movement_state']} | Count: {row['repetition_count']}", (255, 220, 120)),
        (f"Torso-thigh angle: {angle_text} | Side: {row['selected_side'] or 'n/a'}", (245, 245, 245)),
        (f"Tracking: {quality}", color),
    ]
    for index, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (18, 30 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1 if index == 0 else 2,
            cv2.LINE_AA,
        )
    if accepted_number is not None:
        marker = f"REP {accepted_number} ACCEPTED"
        cv2.rectangle(frame, (max(12, width // 2 - 145), height // 2 - 34),
                      (min(width - 12, width // 2 + 145), height // 2 + 18),
                      (25, 145, 35), -1)
        cv2.putText(frame, marker, (max(18, width // 2 - 128), height // 2 + 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def _render_annotated_video(
    video_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    observations: list[Any],
    metadata: Any,
    cv2: Any,
    connections: Any,
    accepted_by_timestamp: dict[int, int],
) -> int:
    capture, render_metadata = open_video(
        video_path, cv2, error_message="Could not reopen input video for rendering"
    )
    writer = None
    try:
        if (
            render_metadata.fps != metadata.fps
            or render_metadata.width != metadata.width
            or render_metadata.height != metadata.height
        ):
            raise RuntimeError("Video metadata changed between inference and rendering")
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create annotated video: {output_path}")

        rendered = 0
        previous_count = 0
        marker_number: int | None = None
        marker_frames = 0
        for frame_index, frame in iter_video_frames(capture):
            if frame_index >= len(rows) or frame_index >= len(observations):
                raise RuntimeError("Video frame count changed during rendering")
            row = rows[frame_index]
            count = int(row["repetition_count"])
            if count > previous_count:
                marker_number = count
                marker_frames = max(1, round(metadata.fps * 0.7))
            points = _pose_points(observations[frame_index])
            draw_normalized_points(frame, points, connections, cv2)
            event_number = accepted_by_timestamp.get(int(row["timestamp_ms"]))
            marker = event_number if event_number is not None else (
                marker_number if marker_frames > 0 else None
            )
            _draw_overlay(
                frame,
                row,
                row.get("visibility_score"),
                marker,
                cv2,
            )
            writer.write(frame)
            rendered += 1
            if marker_frames > 0:
                marker_frames -= 1
            previous_count = count
        if rendered != len(rows):
            raise RuntimeError(f"Rendered {rendered} frames; expected {len(rows)}")
        return rendered
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def _write_angle_plot(path: Path, rows: list[dict[str, Any]], cv2: Any, np: Any) -> None:
    """Write a simple OpenCV plot; missing angles remain visible gaps."""
    width, height = 1200, 720
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 100, width - 55, 55, height - 90
    cv2.putText(canvas, "Curl-up torso-thigh angle vs time (experimental)",
                (left, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    for angle in range(0, 181, 30):
        y = int(bottom - (angle / 180.0) * (bottom - top))
        cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)
        cv2.putText(canvas, str(angle), (left - 42, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (left, bottom), (40, 40, 40), 2)
    cv2.line(canvas, (left, bottom), (right, bottom), (40, 40, 40), 2)
    timestamps = [int(row["timestamp_ms"]) for row in rows]
    max_time = max(timestamps, default=0) / 1000.0
    max_time = max(max_time, 1.0)
    previous: tuple[int, int] | None = None
    for row in rows:
        angle = row["included_angle_deg"]
        if angle is None:
            previous = None
            continue
        value = float(angle)
        if not math.isfinite(value):
            previous = None
            continue
        x = int(left + (int(row["timestamp_ms"]) / 1000.0 / max_time) * (right - left))
        y = int(bottom - (min(180.0, max(0.0, value)) / 180.0) * (bottom - top))
        current = (x, y)
        if previous is not None:
            cv2.line(canvas, previous, current, (205, 75, 35), 2, cv2.LINE_AA)
        cv2.circle(canvas, current, 2, (205, 75, 35), -1, cv2.LINE_AA)
        previous = current
    for second in range(int(math.ceil(max_time)) + 1):
        x = int(left + (second / max_time) * (right - left))
        if x <= right:
            cv2.line(canvas, (x, bottom), (x, bottom + 5), (40, 40, 40), 1)
            cv2.putText(canvas, str(second), (x - 4, bottom + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Time (seconds)", ((left + right) // 2 - 60, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write angle plot: {path}")


def run_diagnostic(
    video_path: Path,
    model_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    video_path = video_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model does not exist: {model_path}")

    output_files = {
        "video": output_dir / "curl_up_annotated.mp4",
        "csv": output_dir / "curl_up_measurements.csv",
        "json": output_dir / "curl_up_summary.json",
        "plot": output_dir / "curl_up_angle_vs_time.png",
    }
    existing = [path for path in output_files.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing diagnostic outputs: "
            + ", ".join(str(path) for path in existing)
            + ". Choose a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    observations: list[Any] = []
    measurements = []
    with PoseVideoPipeline(video_path, model_path) as pipeline:
        metadata = pipeline.metadata
        config = _make_config(args, metadata.width / metadata.height)
        assessment = CurlUpAssessment(config)
        for observation in pipeline:
            observations.append(observation)
            measurements.append(assessment.measure_frame(observation))
        cv2, mp, np = pipeline.cv2, pipeline.mp, pipeline.np

    if not observations:
        raise ValueError(f"Video contains no readable frames: {video_path}")
    result = assessment.analyze(measurements, metadata)
    rows = list(result.summary["measurements"])
    if len(rows) != len(observations):
        raise RuntimeError("Assessment measurements do not match processed frames")
    for row, observation in zip(rows, observations):
        row["pose_detected"] = bool(observation.pose_detected)

    event_list = list(result.events)
    accepted_by_timestamp = {
        int(event["end_timestamp_ms"]): int(event["repetition_number"])
        for event in event_list
        if event["event_type"] == "accepted"
    }
    _write_measurements_csv(
        output_files["csv"],
        rows,
        observations,
        set(accepted_by_timestamp),
    )
    rendered_frames = _render_annotated_video(
        video_path,
        output_files["video"],
        rows,
        observations,
        metadata,
        cv2,
        mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
        accepted_by_timestamp,
    )
    _write_angle_plot(output_files["plot"], rows, cv2, np)

    pose_frames = sum(bool(item.pose_detected) for item in observations)
    valid_frames = sum(row["included_angle_deg"] is not None for row in rows)
    missing_counts = Counter(
        token
        for row in rows
        for token in str(row.get("missing_landmarks", "")).split(",")
        if token
    )
    rejected = [event for event in event_list if event["event_type"] == "rejected"]
    summary: dict[str, Any] = {
        "status": "Experimental diagnostic; not a validated curl-up assessment.",
        "input_video": str(video_path),
        "outputs": {key: str(value) for key, value in output_files.items()},
        "video": {
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "frames_processed": len(observations),
            "frames_rendered": rendered_frames,
        },
        "detector_configuration": result.configuration,
        "detected_repetitions": result.summary["detected_repetitions"],
        "accepted_repetition_timestamps": [
            event for event in event_list if event["event_type"] == "accepted"
        ],
        "rejected_cycles": rejected,
        "quality": {
            "frames_with_detected_poses": pose_frames,
            "pose_detection_percentage": round(
                pose_frames * 100.0 / len(observations), 2
            ),
            "frames_with_valid_angle": valid_frames,
            "angle_measurement_percentage": round(
                valid_frames * 100.0 / len(observations), 2
            ),
            "frames_without_pose": len(observations) - pose_frames,
            "frames_with_invalid_angle": len(observations) - valid_frames,
            "missing_landmark_reasons": dict(missing_counts),
            "selected_side_counts": dict(
                Counter(
                    row["selected_side"]
                    for row in rows
                    if row["selected_side"]
                )
            ),
        },
        "analysis": {
            key: value
            for key, value in result.summary.items()
            if key != "measurements"
        },
        "manual_reference": (
            {
                "approximate_repetitions": args.manual_reference_repetitions,
                "detected_minus_reference": (
                    result.summary["detected_repetitions"]
                    - args.manual_reference_repetitions
                ),
                "interpretation": "Count difference only; not an accuracy estimate.",
            }
            if args.manual_reference_repetitions is not None
            else None
        ),
        "notice": (
            "Thresholds were supplied explicitly and were not tuned to match the "
            "manual reference. The repository contains no curl-up SRS protocol."
        ),
    }
    output_files["json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = run_diagnostic(args.video, args.model, args.output_dir, args)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: diagnostic failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2

    print(f"Experimental repetitions: {summary['detected_repetitions']}")
    for key, path in summary["outputs"].items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
