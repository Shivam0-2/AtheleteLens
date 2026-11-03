"""Extract elbow angles and estimate push-up repetitions from a video.

Elbow angles use normalized image-plane x/y landmarks. Repetition detection is
a configurable threshold state machine and has not been validated against
manually counted repetitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from .athletelens_ai.assessments.push_up import (
        JOINT_NAMES,
        LANDMARK_FIELDS,
        PushUpRepetitionCounter,
        choose_elbow_angle,
        elbow_angle_degrees,
        landmark_row_fields,
        side_measurement,
    )
except ImportError:
    from athletelens_ai.assessments.push_up import (
        JOINT_NAMES,
        LANDMARK_FIELDS,
        PushUpRepetitionCounter,
        choose_elbow_angle,
        elbow_angle_degrees,
        landmark_row_fields,
        side_measurement,
    )

try:
    from .athletelens_ai.pose import (
        create_video_landmarker,
        load_pose_dependencies,
        process_video_frame,
    )
    from .athletelens_ai.video import frame_timestamp_ms, iter_video_frames, open_video
except ImportError:
    from athletelens_ai.pose import (
        create_video_landmarker,
        load_pose_dependencies,
        process_video_frame,
    )
    from athletelens_ai.video import frame_timestamp_ms, iter_video_frames, open_video


def parse_args() -> argparse.Namespace:
    """Parse paths and configurable movement-analysis thresholds."""
    parser = argparse.ArgumentParser(
        description="Measure elbow angles and estimate push-up repetitions."
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video path")
    parser.add_argument("--model", required=True, type=Path, help="Pose .task model")
    parser.add_argument(
        "--csv",
        type=Path,
        dest="csv_path",
        help="Frame-measurement CSV (default: <video>_measurements.csv)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary JSON (default: <video>_summary.json)",
    )
    parser.add_argument(
        "--lower-angle",
        type=float,
        default=90.0,
        help="Bottom threshold in degrees (default: 90)",
    )
    parser.add_argument(
        "--upper-angle",
        type=float,
        default=160.0,
        help="Return-to-top threshold in degrees (default: 160)",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.5,
        help="Minimum supplied landmark visibility (default: 0.5)",
    )
    args = parser.parse_args()
    if not 0 <= args.lower_angle < args.upper_angle <= 180:
        parser.error("angles must satisfy 0 <= --lower-angle < --upper-angle <= 180")
    if not 0 <= args.min_visibility <= 1:
        parser.error("--min-visibility must be between 0 and 1")
    return args


def build_csv_columns() -> list[str]:
    """Define a stable per-frame CSV schema, including raw joint values."""
    columns = [
        "frame_number",
        "timestamp_ms",
        "pose_detected",
        "selected_side",
        "elbow_angle_deg",
        "left_elbow_angle_deg",
        "right_elbow_angle_deg",
    ]
    for side in ("left", "right"):
        for joint in JOINT_NAMES:
            columns.extend(
                f"{side}_{joint}_{field}" for field in LANDMARK_FIELDS
            )
    return columns


def analyze_video(
    video_path: Path,
    model_path: Path,
    csv_path: Path,
    summary_path: Path,
    lower_angle: float,
    upper_angle: float,
    min_visibility: float,
) -> dict[str, Any]:
    """Process each frame, write raw measurements, then write a summary."""
    cv2, mp, np = load_pose_dependencies()

    capture, metadata = open_video(
        video_path, cv2, error_message="Could not open input video"
    )

    fps = metadata.fps
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Input video has invalid FPS metadata: {fps}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    counter = PushUpRepetitionCounter(lower_angle, upper_angle)
    valid_angles: list[float] = []
    pose_frames = 0
    processed_frames = 0
    previous_timestamp_ms = -1
    csv_columns = build_csv_columns()

    try:
        try:
            landmarker_context = create_video_landmarker(mp, str(model_path))
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Could not initialize Pose Landmarker from model '{model_path}'. "
                "Check that it is a valid compatible .task model."
            ) from exc

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()

            with landmarker_context as landmarker:
                for frame_index, frame_bgr in iter_video_frames(capture):
                    frame_number = processed_frames + 1
                    timestamp_ms = frame_timestamp_ms(
                        frame_index, fps, previous_timestamp_ms
                    )
                    previous_timestamp_ms = timestamp_ms
                    result = process_video_frame(
                        frame_bgr, timestamp_ms, landmarker, cv2, mp, np
                    )

                    angle = None
                    selected_side = ""
                    landmark_fields = {
                        f"{side}_{joint}_{field}": ""
                        for side in ("left", "right")
                        for joint in JOINT_NAMES
                        for field in LANDMARK_FIELDS
                    }
                    left_angle = right_angle = None
                    if result.pose_landmarks:
                        pose_frames += 1
                        pose = result.pose_landmarks[0]
                        (
                            angle,
                            selected_side,
                            left_angle,
                            right_angle,
                            landmark_fields,
                        ) = choose_elbow_angle(pose, min_visibility)

                    counter.update(angle)
                    if angle is not None:
                        valid_angles.append(angle)

                    writer.writerow(
                        {
                            "frame_number": frame_number,
                            "timestamp_ms": timestamp_ms,
                            "pose_detected": bool(result.pose_landmarks),
                            "selected_side": selected_side,
                            "elbow_angle_deg": round(angle, 3) if angle is not None else "",
                            "left_elbow_angle_deg": (
                                round(left_angle, 3) if left_angle is not None else ""
                            ),
                            "right_elbow_angle_deg": (
                                round(right_angle, 3) if right_angle is not None else ""
                            ),
                            **landmark_fields,
                        }
                    )
                    processed_frames += 1

        if processed_frames == 0:
            raise ValueError(f"Input video contains no readable frames: {video_path}")

        summary = {
            "input_video": str(video_path),
            "total_frames_processed": processed_frames,
            "frames_with_detected_poses": pose_frames,
            "frames_with_valid_elbow_angle": len(valid_angles),
            "total_detected_repetitions": counter.repetitions,
            "thresholds_degrees": {
                "lower_angle": lower_angle,
                "upper_angle": upper_angle,
            },
            "average_elbow_angle_degrees": (
                round(sum(valid_angles) / len(valid_angles), 3) if valid_angles else None
            ),
            "minimum_elbow_angle_degrees": (
                round(min(valid_angles), 3) if valid_angles else None
            ),
            "maximum_elbow_angle_degrees": (
                round(max(valid_angles), 3) if valid_angles else None
            ),
            "repetition_count_validation": (
                "Prototype threshold estimate; not validated against manual counts."
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        capture.release()


def main() -> int:
    """Validate paths, run analysis, and report the output locations."""
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    csv_path = (
        args.csv_path.expanduser().resolve()
        if args.csv_path
        else video_path.with_name(f"{video_path.stem}_measurements.csv")
    )
    summary_path = (
        args.summary.expanduser().resolve()
        if args.summary
        else video_path.with_name(f"{video_path.stem}_summary.json")
    )

    if not video_path.is_file():
        print(f"Error: input video does not exist: {video_path}", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"Error: model file does not exist: {model_path}", file=sys.stderr)
        return 2
    if csv_path == video_path or summary_path == video_path:
        print("Error: output paths must not overwrite the input video.", file=sys.stderr)
        return 2
    if csv_path == summary_path:
        print("Error: CSV and summary paths must be different.", file=sys.stderr)
        return 2

    try:
        summary = analyze_video(
            video_path,
            model_path,
            csv_path,
            summary_path,
            args.lower_angle,
            args.upper_angle,
            args.min_visibility,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"Error: analysis failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 2

    print(json.dumps(summary, indent=2))
    print(f"Frame measurements: {csv_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
