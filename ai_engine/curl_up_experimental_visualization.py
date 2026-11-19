"""Render the experimental 115/100 curl-up configuration on the verified clip.

Pose inference and timestamps come from the shared pipeline. This runner is
pinned to ``videos/curl up.mp4`` so it cannot silently switch to a similarly
named input. It does not change the curl-up state machine or its defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Any

from athletelens_ai.assessments.curl_up import CurlUpAssessment, CurlUpConfig
from athletelens_ai.pipeline import PoseVideoPipeline
from athletelens_ai.video import iter_video_frames, open_video
from athletelens_ai.visualization import draw_normalized_points


AI_ENGINE_DIR = Path(__file__).resolve().parent
SOURCE_VIDEO = AI_ENGINE_DIR / "videos" / "curl up.mp4"
DEFAULT_MODEL = AI_ENGINE_DIR / "models" / "pose_landmarker_lite.task"
DEFAULT_OUTPUT_DIR = (
    AI_ENGINE_DIR
    / "experimental_outputs"
    / "curl_up_diagnostic"
    / "visualization_115_100_real_video_v2"
)

EXPERIMENT_CONFIG = {
    "extended_angle_deg": 115.0,
    "flexed_angle_deg": 100.0,
    "hysteresis_deg": 5.0,
    "confirmation_frames": 2,
    "min_repetition_duration_s": 0.2,
    "max_repetition_duration_s": 3.0,
    "max_measurement_gap_s": 0.25,
    "min_visibility": 0.5,
    "starting_position": "extended",
    "selected_side": "auto",
    "test_duration_s": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an experimental curl-up visualization for the exact source "
            "videos/curl up.mp4. The input video is intentionally fixed."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _points(observation: Any) -> list[tuple[float, float] | None]:
    if not observation.pose_landmark_data:
        return []
    points: list[tuple[float, float] | None] = []
    for landmark in observation.pose_landmark_data[0]:
        try:
            x, y = float(landmark["x"]), float(landmark["y"])
        except (KeyError, TypeError, ValueError):
            points.append(None)
            continue
        points.append((x, y) if math.isfinite(x) and math.isfinite(y) else None)
    return points


def _visibility(observation: Any, selected_side: str) -> float | None:
    if selected_side not in {"left", "right"} or not observation.pose_landmark_data:
        return None
    joint_indexes = {
        "shoulder": 11 if selected_side == "left" else 12,
        "hip": 23 if selected_side == "left" else 24,
        "knee": 25 if selected_side == "left" else 26,
    }
    scores = []
    pose = observation.pose_landmark_data[0]
    for index in joint_indexes.values():
        if index >= len(pose):
            return None
        try:
            score = float(pose[index]["visibility"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(score):
            return None
        scores.append(score)
    return min(scores)


def _event_frame_index(event: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    timestamps = [int(row["timestamp_ms"]) for row in rows]
    index = bisect_left(timestamps, int(event["end_timestamp_ms"]))
    if index >= len(timestamps):
        return len(timestamps) - 1
    if index and abs(timestamps[index - 1] - event["end_timestamp_ms"]) <= abs(
        timestamps[index] - event["end_timestamp_ms"]
    ):
        return index - 1
    return index


def _accepted_event_frames(
    events: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    """Place acceptance markers on the frame where the accepted count changes.

    The detector records an event at the first confirmation sample, then accepts
    it after the confirmation sequence completes. The display belongs on that
    latter frame while retaining the detector's event timestamp in the label.
    """
    mapped: dict[int, dict[str, Any]] = {}
    for event in events:
        target_count = int(event["repetition_number"])
        frame_index = next(
            (
                index
                for index, row in enumerate(rows)
                if int(row["repetition_count"]) >= target_count
            ),
            None,
        )
        if frame_index is None:
            frame_index = _event_frame_index(event, rows)
        mapped[frame_index] = {
            **event,
            "accepted_on_frame_number": rows[frame_index]["frame_number"],
            "accepted_on_timestamp_ms": rows[frame_index]["timestamp_ms"],
        }
    return mapped


def _draw_panel(
    frame: Any,
    row: dict[str, Any],
    visibility: float | None,
    cv2: Any,
    accepted_event: dict[str, Any] | None = None,
    rejected_event: dict[str, Any] | None = None,
) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (min(width - 8, 515), 138), (20, 25, 33), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    angle = row.get("included_angle_deg")
    angle_text = f"{float(angle):.1f} deg" if angle is not None else "unavailable"
    if not row.get("pose_detected", False):
        tracking = "NO POSE"
        tracking_color = (70, 75, 235)
    elif not row.get("measurement_valid", False):
        tracking = "LANDMARKS / ANGLE INVALID"
        tracking_color = (20, 160, 245)
    else:
        score = "n/a" if visibility is None else f"{visibility:.2f}"
        tracking = f"TRACKED (visibility {score})"
        tracking_color = (60, 220, 100)

    lines = [
        ("AthleteLens | EXPERIMENTAL / UNVALIDATED", (245, 245, 245)),
        (
            f"State: {row['movement_state']} | Repetitions: {row['repetition_count']}",
            (255, 220, 115),
        ),
        (
            f"Included angle: {angle_text} | Side: {row.get('selected_side') or 'n/a'}",
            (245, 245, 245),
        ),
        (f"Tracking: {tracking}", tracking_color),
    ]
    for index, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (18, 30 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            color,
            1 if index == 0 else 2,
            cv2.LINE_AA,
        )

    if accepted_event is not None:
        timestamp_s = int(accepted_event["end_timestamp_ms"]) / 1000.0
        marker = f"REP {accepted_event['repetition_number']} ACCEPTED | EVENT {timestamp_s:.3f}s"
        _draw_event_banner(frame, marker, (25, 145, 35), cv2)
    elif rejected_event is not None:
        timestamp_s = int(rejected_event["end_timestamp_ms"]) / 1000.0
        marker = f"REJECTED: {rejected_event['reason']} @ {timestamp_s:.3f}s"
        _draw_event_banner(frame, marker, (25, 70, 190), cv2)


def _draw_event_banner(frame: Any, text: str, color: tuple[int, int, int], cv2: Any) -> None:
    height, width = frame.shape[:2]
    scale, thickness = 0.7, 2
    (text_width, text_height), _baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness
    )
    x = max(8, (width - text_width) // 2)
    y = max(160, height // 2)
    cv2.rectangle(
        frame,
        (x - 10, y - text_height - 12),
        (min(width - 8, x + text_width + 10), y + 10),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _render_video(
    output_path: Path,
    rows: list[dict[str, Any]],
    observations: list[Any],
    metadata: Any,
    cv2: Any,
    connections: Any,
    accepted_frame_events: dict[int, dict[str, Any]],
    rejected_frame_events: dict[int, dict[str, Any]],
) -> int:
    capture, render_metadata = open_video(
        SOURCE_VIDEO, cv2, error_message="Could not reopen the verified source video"
    )
    writer = None
    try:
        if (
            render_metadata.fps != metadata.fps
            or render_metadata.width != metadata.width
            or render_metadata.height != metadata.height
        ):
            raise RuntimeError("Source metadata changed between inference and rendering")
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create annotated MP4: {output_path}")

        rendered = 0
        active_accepted: dict[str, Any] | None = None
        active_rejected: dict[str, Any] | None = None
        accepted_frames_left = rejected_frames_left = 0
        marker_duration = max(1, round(metadata.fps * 0.9))
        for frame_index, frame in iter_video_frames(capture):
            if frame_index >= len(rows) or frame_index >= len(observations):
                raise RuntimeError("Source frame count changed during rendering")
            row = rows[frame_index]
            if frame_index in accepted_frame_events:
                active_accepted = accepted_frame_events[frame_index]
                accepted_frames_left = marker_duration
                active_rejected = None
                rejected_frames_left = 0
            elif frame_index in rejected_frame_events:
                active_rejected = rejected_frame_events[frame_index]
                rejected_frames_left = marker_duration
                active_accepted = None
                accepted_frames_left = 0

            draw_normalized_points(frame, _points(observations[frame_index]), connections, cv2)
            marker_accepted = (
                active_accepted
                if active_accepted is not None and accepted_frames_left > 0
                else None
            )
            marker_rejected = (
                active_rejected
                if active_rejected is not None and rejected_frames_left > 0
                else None
            )
            _draw_panel(
                frame,
                row,
                _visibility(observations[frame_index], row.get("selected_side", "")),
                cv2,
                marker_accepted,
                marker_rejected,
            )
            writer.write(frame)
            rendered += 1
            accepted_frames_left = max(0, accepted_frames_left - 1)
            rejected_frames_left = max(0, rejected_frames_left - 1)

        if rendered != len(rows):
            raise RuntimeError(f"Rendered {rendered} frames; expected {len(rows)}")
        return rendered
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def run_visualization(model_path: Path, output_dir: Path) -> dict[str, Any]:
    source = SOURCE_VIDEO.resolve()
    model_path = model_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Verified source video is missing: {source}. No alternate input will be substituted."
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model is missing: {model_path}")

    output_video = output_dir / "curl_up_115_100_annotated.mp4"
    output_json = output_dir / "curl_up_115_100_events.json"
    existing = [path for path in (output_video, output_json) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; choose a new --output-dir: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    observations: list[Any] = []
    measurements = []
    with PoseVideoPipeline(source, model_path) as pipeline:
        metadata = pipeline.metadata
        config = CurlUpConfig(
            image_aspect_ratio=metadata.width / metadata.height,
            **EXPERIMENT_CONFIG,
        )
        assessment = CurlUpAssessment(config)
        for observation in pipeline:
            observations.append(observation)
            measurements.append(assessment.measure_frame(observation))
        cv2, mp, _np = pipeline.cv2, pipeline.mp, pipeline.np

    if not observations:
        raise ValueError(f"The verified source contains no readable frames: {source}")
    result = assessment.analyze(measurements, metadata)
    rows = list(result.summary["measurements"])
    if len(rows) != len(observations):
        raise RuntimeError("Assessment output does not align with source frames")
    for row, observation in zip(rows, observations):
        row["pose_detected"] = bool(observation.pose_detected)

    accepted = [event for event in result.events if event["event_type"] == "accepted"]
    rejected = [event for event in result.events if event["event_type"] == "rejected"]
    accepted_frame_events = _accepted_event_frames(accepted, rows)
    rejected_frame_events = {
        _event_frame_index(event, rows): event for event in rejected
    }
    rendered = _render_video(
        output_video,
        rows,
        observations,
        metadata,
        cv2,
        mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
        accepted_frame_events,
        rejected_frame_events,
    )

    pose_count = sum(bool(observation.pose_detected) for observation in observations)
    valid_angle_count = sum(row["included_angle_deg"] is not None for row in rows)
    summary = {
        "status": "Experimental visualization; not a validated assessment.",
        "source_video": str(source),
        "source_video_size_bytes": source.stat().st_size,
        "source_video_sha256": _sha256(source),
        "video": {
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "frames_processed": len(observations),
            "frames_rendered": rendered,
        },
        "detector_configuration": result.configuration,
        "detected_repetitions": result.summary["detected_repetitions"],
        "accepted_repetition_timestamps": list(accepted_frame_events.values()),
        "rejected_cycles": rejected,
        "quality": {
            "frames_with_detected_poses": pose_count,
            "pose_detection_percentage": round(
                pose_count * 100.0 / len(observations), 2
            ),
            "frames_with_valid_included_angle": valid_angle_count,
            "angle_measurement_percentage": round(
                valid_angle_count * 100.0 / len(observations), 2
            ),
            "average_included_angle_deg": result.summary[
                "average_included_angle_deg"
            ],
        },
        "outputs": {
            "annotated_video": str(output_video),
            "event_summary": str(output_json),
        },
        "notice": (
            "This experimental configuration uses 115/100 degree thresholds. "
            "The approximate manual reference is not used to tune the detector. "
            "One video is insufficient to validate accuracy."
        ),
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    try:
        result = run_visualization(args.model, args.output_dir)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: visualization failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2

    print(f"Source: {result['source_video']}")
    print(f"Experimental repetitions: {result['detected_repetitions']}")
    print(f"Annotated MP4: {result['outputs']['annotated_video']}")
    print(f"JSON summary: {result['outputs']['event_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
