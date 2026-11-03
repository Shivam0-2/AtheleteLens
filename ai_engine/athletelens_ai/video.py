"""Shared OpenCV video metadata, frame iteration, and timestamp helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class VideoMetadata:
    """Video properties read from an OpenCV capture."""

    fps: float
    width: int
    height: int


def open_video(
    video_path: str | Path,
    cv2_module: Any | None = None,
    error_message: str = "Could not open video",
) -> tuple[Any, VideoMetadata]:
    """Open a video with OpenCV and return its capture and source metadata."""
    if cv2_module is None:
        import cv2 as cv2_module

    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"{error_message}: {video_path}")
    metadata = VideoMetadata(
        fps=float(capture.get(cv2_module.CAP_PROP_FPS)),
        width=int(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT)),
    )
    return capture, metadata


def iter_video_frames(capture: Any) -> Iterator[tuple[int, Any]]:
    """Yield decoded frames with zero-based indices until OpenCV reaches EOF."""
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        yield frame_index, frame
        frame_index += 1


def frame_timestamp_ms(
    frame_index: int, fps: float, previous_timestamp_ms: int = -1
) -> int:
    """Return the existing monotonic, frame-rate-derived timestamp in ms."""
    return max(
        previous_timestamp_ms + 1,
        int(round(frame_index * 1000.0 / fps)),
    )
