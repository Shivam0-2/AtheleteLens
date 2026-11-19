"""Shared video-to-pose iteration for assessment code.

This module owns one video capture and one MediaPipe VIDEO landmarker for the
duration of a pass. It deliberately contains no assessment or web-framework
logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .pose import (
    create_video_landmarker,
    extract_landmark_sets,
    load_pose_dependencies,
    process_video_frame,
)
from .video import (
    VideoMetadata,
    frame_timestamp_ms,
    iter_video_frames,
    open_video,
)


class VideoInputError(ValueError):
    """The video could not be opened or does not have usable metadata."""


@dataclass(frozen=True)
class PoseObservation:
    """Pose output for one decoded video frame.

    ``pose_landmarks`` and ``pose_world_landmarks`` retain the MediaPipe
    landmark objects for assessment calculations. Their serialized companions
    preserve only fields actually supplied by MediaPipe; an empty pose list
    records a frame where no pose was detected.
    """

    frame_number: int
    timestamp_ms: int
    pose_landmarks: list[list[Any]]
    pose_world_landmarks: list[list[Any]]
    pose_landmark_data: list[list[dict[str, Any]]]
    pose_world_landmark_data: list[list[dict[str, Any]]]
    result: Any

    @property
    def pose_detected(self) -> bool:
        """Whether MediaPipe returned at least one pose for this frame."""
        return bool(self.pose_landmarks)


class PoseVideoPipeline:
    """Context-managed, sequential video-to-pose processor.

    Optional helper arguments are dependency seams for deterministic tests and
    callers that already wrap errors. Normal callers should use the defaults.
    """

    def __init__(
        self,
        video_path: str | Path,
        model_path: str | Path,
        *,
        dependency_loader: Callable[[], tuple[Any, Any, Any]] = load_pose_dependencies,
        video_opener: Callable[..., tuple[Any, VideoMetadata]] = open_video,
        landmarker_factory: Callable[..., Any] = create_video_landmarker,
        frame_iterator: Callable[[Any], Iterator[tuple[int, Any]]] = iter_video_frames,
        timestamp_factory: Callable[[int, float, int], int] = frame_timestamp_ms,
        frame_processor: Callable[..., Any] = process_video_frame,
    ) -> None:
        self.video_path = Path(video_path)
        self.model_path = Path(model_path)
        self._dependency_loader = dependency_loader
        self._video_opener = video_opener
        self._landmarker_factory = landmarker_factory
        self._frame_iterator = frame_iterator
        self._timestamp_factory = timestamp_factory
        self._frame_processor = frame_processor
        self.cv2: Any = None
        self.mp: Any = None
        self.np: Any = None
        self.capture: Any = None
        self.metadata: VideoMetadata | None = None
        self._landmarker_context: Any = None
        self.landmarker: Any = None
        self._previous_timestamp_ms = -1
        self._entered = False

    def __enter__(self) -> PoseVideoPipeline:
        try:
            self.cv2, self.mp, self.np = self._dependency_loader()
            try:
                self.capture, self.metadata = self._video_opener(
                    self.video_path, self.cv2
                )
            except (OSError, ValueError) as exc:
                raise VideoInputError(
                    f"Could not open video: {self.video_path}"
                ) from exc

            if (
                not math.isfinite(self.metadata.fps)
                or self.metadata.fps <= 0
                or self.metadata.width <= 0
                or self.metadata.height <= 0
            ):
                raise VideoInputError("Video metadata is invalid")

            self._landmarker_context = self._landmarker_factory(
                self.mp, str(self.model_path)
            )
            if hasattr(self._landmarker_context, "__enter__"):
                self.landmarker = self._landmarker_context.__enter__()
            else:
                self.landmarker = self._landmarker_context
            self._entered = True
            return self
        except Exception:
            self._cleanup()
            raise

    def __iter__(self) -> Iterator[PoseObservation]:
        if not self._entered or self.capture is None or self.metadata is None:
            raise RuntimeError("PoseVideoPipeline must be entered before iteration")

        for frame_number, frame_bgr in self._frame_iterator(self.capture):
            timestamp_ms = self._timestamp_factory(
                frame_number, self.metadata.fps, self._previous_timestamp_ms
            )
            if timestamp_ms <= self._previous_timestamp_ms:
                timestamp_ms = self._previous_timestamp_ms + 1
            self._previous_timestamp_ms = timestamp_ms
            result = self._frame_processor(
                frame_bgr,
                timestamp_ms,
                self.landmarker,
                self.cv2,
                self.mp,
                self.np,
            )
            poses = getattr(result, "pose_landmarks", None) or []
            world_poses = getattr(result, "pose_world_landmarks", None) or []
            yield PoseObservation(
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                pose_landmarks=poses,
                pose_world_landmarks=world_poses,
                pose_landmark_data=extract_landmark_sets(
                    result, "pose_landmarks"
                ),
                pose_world_landmark_data=extract_landmark_sets(
                    result, "pose_world_landmarks"
                ),
                result=result,
            )

    def _cleanup(self) -> None:
        """Release both native resources, even if one cleanup action fails."""
        capture, self.capture = self.capture, None
        context, self._landmarker_context = self._landmarker_context, None
        landmarker, self.landmarker = self.landmarker, None
        try:
            if capture is not None:
                capture.release()
        finally:
            if context is not None:
                if hasattr(context, "__exit__"):
                    context.__exit__(None, None, None)
                elif landmarker is not None and hasattr(landmarker, "close"):
                    landmarker.close()
            self._entered = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._cleanup()

