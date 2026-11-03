"""Render MediaPipe Pose Landmarker landmarks and connections onto a video."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

try:
    from .athletelens_ai.pose import (
        create_video_landmarker,
        load_pose_dependencies,
        process_video_frame,
    )
    from .athletelens_ai.video import frame_timestamp_ms, iter_video_frames, open_video
    from .athletelens_ai.visualization import draw_pose_landmarks
except ImportError:
    from athletelens_ai.pose import (
        create_video_landmarker,
        load_pose_dependencies,
        process_video_frame,
    )
    from athletelens_ai.video import frame_timestamp_ms, iter_video_frames, open_video
    from athletelens_ai.visualization import draw_pose_landmarks


def parse_args() -> argparse.Namespace:
    """Parse paths for the input video, task model, and rendered output."""
    parser = argparse.ArgumentParser(
        description="Create a video with MediaPipe pose landmarks overlaid."
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video path")
    parser.add_argument("--model", required=True, type=Path, help="Pose .task model")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 path (default: pushup_annotated.mp4 beside the input)",
    )
    return parser.parse_args()


def render_video(video_path: Path, model_path: Path, output_path: Path) -> int:
    """Process every input frame and write an annotated MP4; return frame count."""
    cv2, mp, np = load_pose_dependencies()

    capture, metadata = open_video(
        video_path, cv2, error_message="Could not open input video"
    )

    writer = None
    try:
        fps = metadata.fps
        width = metadata.width
        height = metadata.height
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                f"Input video has invalid metadata (FPS={fps}, "
                f"dimensions={width}x{height}): {video_path}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(
                f"Could not create MP4 output at '{output_path}'. "
                "Check the output path and OpenCV video codec support."
            )

        try:
            landmarker_context = create_video_landmarker(mp, str(model_path))
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Could not initialize Pose Landmarker from model '{model_path}'. "
                "Check that it is a valid compatible .task model."
            ) from exc

        pose_connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        drawing = mp.tasks.vision.drawing_utils
        landmark_spec = drawing.DrawingSpec(
            color=(0, 255, 255), thickness=2, circle_radius=3
        )
        connection_spec = drawing.DrawingSpec(
            color=(0, 220, 0), thickness=2, circle_radius=2
        )

        processed_frames = 0
        previous_timestamp_ms = -1
        with landmarker_context as landmarker:
            for frame_index, frame_bgr in iter_video_frames(capture):
                timestamp_ms = frame_timestamp_ms(
                    frame_index, fps, previous_timestamp_ms
                )
                previous_timestamp_ms = timestamp_ms
                result = process_video_frame(
                    frame_bgr, timestamp_ms, landmarker, cv2, mp, np
                )

                for pose_landmarks in result.pose_landmarks:
                    draw_pose_landmarks(
                        frame_bgr,
                        pose_landmarks,
                        pose_connections,
                        drawing,
                        landmark_spec,
                        connection_spec,
                    )

                # With no poses, this writes the original frame unchanged.
                writer.write(frame_bgr)
                processed_frames += 1

        if processed_frames == 0:
            raise ValueError(f"Input video contains no readable frames: {video_path}")
        return processed_frames
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main() -> int:
    """Validate input files and render the annotated video."""
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else video_path.with_name("pushup_annotated.mp4")
    )

    if not video_path.is_file():
        print(f"Error: input video does not exist: {video_path}", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"Error: model file does not exist: {model_path}", file=sys.stderr)
        return 2
    if output_path == video_path:
        print("Error: output path must not overwrite the input video.", file=sys.stderr)
        return 2

    try:
        frame_count = render_video(video_path, model_path, output_path)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"Error: rendering failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 2

    print(f"Annotated video saved: {output_path}")
    print(f"Frames processed: {frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
