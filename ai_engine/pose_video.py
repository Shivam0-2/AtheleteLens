"""Run MediaPipe Pose Landmarker over a video and report raw landmarks.

This prototype reports detections and the first few detected frames only. It
does not classify exercises, count repetitions, or calculate performance scores.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from .athletelens_ai.pose import (
        available_landmark_data,
        create_video_landmarker,
        extract_landmark_sets,
        load_pose_dependencies,
        process_video_frame,
    )
    from .athletelens_ai.video import frame_timestamp_ms, iter_video_frames, open_video
except ImportError:
    from athletelens_ai.pose import (
        available_landmark_data,
        create_video_landmarker,
        extract_landmark_sets,
        load_pose_dependencies,
        process_video_frame,
    )
    from athletelens_ai.video import frame_timestamp_ms, iter_video_frames, open_video


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Pose Landmarker landmarks from a video."
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video path")
    parser.add_argument(
        "--model", required=True, type=Path, help="Pose Landmarker .task model path"
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=3,
        help="Maximum detected frames whose landmark data is printed (default: 3)",
    )
    args = parser.parse_args()
    if args.sample_frames < 0:
        parser.error("--sample-frames must be zero or greater")
    return args


def run(video_path: Path, model_path: Path, sample_frame_limit: int) -> dict[str, Any]:
    """Process the video sequentially and return detection statistics."""
    cv2, mp, np = load_pose_dependencies()

    capture, metadata = open_video(video_path, cv2)

    fps = metadata.fps
    width = metadata.width
    height = metadata.height
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(
            f"Video metadata is invalid (FPS={fps}, dimensions={width}x{height}): "
            f"{video_path}"
        )

    try:
        try:
            landmarker_context = create_video_landmarker(mp, str(model_path))
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Could not initialize Pose Landmarker from model '{model_path}'. "
                "Check that it is a valid compatible .task model."
            ) from exc

        processed_frames = 0
        frames_with_poses = 0
        previous_timestamp_ms = -1
        sample_frames: list[dict[str, Any]] = []

        with landmarker_context as landmarker:
            for frame_index, frame_bgr in iter_video_frames(capture):
                # Derive time from the media rate; clamp to guarantee strictly
                # increasing integer timestamps even for unusual FPS metadata.
                timestamp_ms = frame_timestamp_ms(
                    frame_index, fps, previous_timestamp_ms
                )
                previous_timestamp_ms = timestamp_ms
                result = process_video_frame(
                    frame_bgr, timestamp_ms, landmarker, cv2, mp, np
                )
                frame_poses = extract_landmark_sets(result, "pose_landmarks")
                frame_world_poses = extract_landmark_sets(
                    result, "pose_world_landmarks"
                )

                processed_frames += 1
                if frame_poses:
                    frames_with_poses += 1
                    if len(sample_frames) < sample_frame_limit:
                        sample_frames.append(
                            {
                                "frame_index": frame_index,
                                "timestamp_ms": timestamp_ms,
                                "pose_landmarks": frame_poses,
                                "pose_world_landmarks": frame_world_poses,
                            }
                        )

        if processed_frames == 0:
            raise ValueError(f"Video contains no readable frames: {video_path}")

        return {
            "video": str(video_path),
            "fps": fps,
            "dimensions": {"width": width, "height": height},
            "total_frames_processed": processed_frames,
            "frames_with_detected_poses": frames_with_poses,
            "pose_detection_percentage": round(
                frames_with_poses * 100.0 / processed_frames, 2
            ),
            "sample_frames": sample_frames,
        }
    finally:
        capture.release()


def main() -> int:
    """Validate inputs, run extraction, and print a JSON report."""
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
        report = run(video_path, model_path, args.sample_frames)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Native CV/MediaPipe failures should be actionable.
        print(
            f"Error: video processing failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 2

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
