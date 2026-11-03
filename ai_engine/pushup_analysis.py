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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JOINT_NAMES = ("shoulder", "elbow", "wrist")
LANDMARK_FIELDS = ("x", "y", "z", "visibility", "presence")


@dataclass
class PushUpRepetitionCounter:
    """Count a repetition after a lower-threshold then upper-threshold crossing."""

    lower_angle_deg: float
    upper_angle_deg: float
    repetitions: int = 0
    phase: str = "seeking_bottom"

    def update(self, elbow_angle_deg: float | None) -> int:
        """Update the state from one angle; missing angles do not change state."""
        if elbow_angle_deg is None:
            return self.repetitions
        if self.phase == "seeking_bottom" and elbow_angle_deg <= self.lower_angle_deg:
            self.phase = "bottom_reached"
        elif self.phase == "bottom_reached" and elbow_angle_deg >= self.upper_angle_deg:
            self.repetitions += 1
            self.phase = "top_reached"
        elif self.phase == "top_reached" and elbow_angle_deg <= self.lower_angle_deg:
            self.phase = "bottom_reached"
        return self.repetitions


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


def elbow_angle_degrees(shoulder: Any, elbow: Any, wrist: Any) -> float | None:
    """Compute the 2D shoulder-elbow-wrist angle in degrees, or None if invalid."""
    try:
        shoulder_xy = (float(shoulder.x), float(shoulder.y))
        elbow_xy = (float(elbow.x), float(elbow.y))
        wrist_xy = (float(wrist.x), float(wrist.y))
    except (AttributeError, TypeError, ValueError):
        return None

    upper_arm = (shoulder_xy[0] - elbow_xy[0], shoulder_xy[1] - elbow_xy[1])
    forearm = (wrist_xy[0] - elbow_xy[0], wrist_xy[1] - elbow_xy[1])
    upper_length = math.hypot(*upper_arm)
    forearm_length = math.hypot(*forearm)
    if upper_length == 0 or forearm_length == 0:
        return None

    cosine = (
        upper_arm[0] * forearm[0] + upper_arm[1] * forearm[1]
    ) / (upper_length * forearm_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def landmark_row_fields(prefix: str, landmark: Any | None) -> dict[str, float | str]:
    """Return the supplied landmark fields for one joint, blank if unavailable."""
    values: dict[str, float | str] = {}
    for field in LANDMARK_FIELDS:
        value = getattr(landmark, field, None) if landmark is not None else None
        values[f"{prefix}_{field}"] = float(value) if value is not None else ""
    return values


def side_measurement(
    landmarks: list[Any], side: str, min_visibility: float
) -> tuple[float | None, dict[str, float | str], float | None]:
    """Extract a side's shoulder/elbow/wrist and compute its visible angle."""
    import mediapipe as mp

    pose_landmark = mp.tasks.vision.PoseLandmark
    indices = {
        "shoulder": getattr(pose_landmark, f"{side.upper()}_SHOULDER").value,
        "elbow": getattr(pose_landmark, f"{side.upper()}_ELBOW").value,
        "wrist": getattr(pose_landmark, f"{side.upper()}_WRIST").value,
    }
    joints: dict[str, Any | None] = {
        joint: landmarks[index] if index < len(landmarks) else None
        for joint, index in indices.items()
    }
    fields: dict[str, float | str] = {}
    for joint, landmark in joints.items():
        fields.update(landmark_row_fields(f"{side}_{joint}", landmark))

    confidences = [
        float(landmark.visibility)
        for landmark in joints.values()
        if landmark is not None and getattr(landmark, "visibility", None) is not None
    ]
    visibility_score = min(confidences) if len(confidences) == len(JOINT_NAMES) else None
    visibility_ok = all(
        landmark is not None
        and (
            getattr(landmark, "visibility", None) is None
            or float(landmark.visibility) >= min_visibility
        )
        for landmark in joints.values()
    )
    angle = (
        elbow_angle_degrees(joints["shoulder"], joints["elbow"], joints["wrist"])
        if visibility_ok
        else None
    )
    return angle, fields, visibility_score


def choose_elbow_angle(
    landmarks: list[Any], min_visibility: float
) -> tuple[float | None, str, float | None, float | None, dict[str, float | str]]:
    """Choose the valid side with the strongest visibility; prefer left on ties."""
    left_angle, left_fields, left_score = side_measurement(
        landmarks, "left", min_visibility
    )
    right_angle, right_fields, right_score = side_measurement(
        landmarks, "right", min_visibility
    )
    fields = {**left_fields, **right_fields}

    candidates = [
        ("left", left_angle, left_score),
        ("right", right_angle, right_score),
    ]
    candidates = [candidate for candidate in candidates if candidate[1] is not None]
    if not candidates:
        return None, "", left_angle, right_angle, fields
    # Scores are None only when the model did not supply visibility values.
    side, angle, _ = max(
        candidates,
        key=lambda candidate: (
            candidate[2] if candidate[2] is not None else -1.0,
            candidate[0] == "left",
        ),
    )
    return angle, side, left_angle, right_angle, fields


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
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "A required dependency is unavailable. Activate the ai_engine "
            "virtual environment and run: python -m pip install -r requirements.txt"
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open input video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
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
                f"Could not initialize Pose Landmarker from model '{model_path}'. "
                "Check that it is a valid compatible .task model."
            ) from exc

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()

            with landmarker_context as landmarker:
                while True:
                    success, frame_bgr = capture.read()
                    if not success:
                        break

                    frame_number = processed_frames + 1
                    timestamp_ms = max(
                        previous_timestamp_ms + 1,
                        int(round(processed_frames * 1000.0 / fps)),
                    )
                    previous_timestamp_ms = timestamp_ms
                    frame_rgb = np.ascontiguousarray(
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    )
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb),
                        timestamp_ms,
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
