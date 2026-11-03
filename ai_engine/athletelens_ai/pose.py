"""Shared MediaPipe Pose Landmarker setup and result serialization."""

from __future__ import annotations

from typing import Any


def load_pose_dependencies() -> tuple[Any, Any, Any]:
    """Load OpenCV, MediaPipe, and NumPy with one actionable setup error."""
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "A required dependency is unavailable. Activate the ai_engine "
            "virtual environment and run: python -m pip install -r requirements.txt"
        ) from exc
    return cv2, mp, np


def create_video_landmarker(
    mp: Any, model_path: str, num_poses: int = 1
) -> Any:
    """Create a Pose Landmarker configured for sequential VIDEO processing."""
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=num_poses,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def process_video_frame(
    frame_bgr: Any,
    timestamp_ms: int,
    landmarker: Any,
    cv2: Any,
    mp: Any,
    np: Any,
) -> Any:
    """Convert one OpenCV BGR frame to MediaPipe RGB and detect its pose."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def available_landmark_data(landmark: Any) -> dict[str, float]:
    """Return coordinate and confidence values supplied by a landmark."""
    values: dict[str, float] = {}
    for field in ("x", "y", "z", "visibility", "presence"):
        value = getattr(landmark, field, None)
        if value is not None:
            values[field] = float(value)
    return values


def extract_landmark_sets(result: Any, attribute: str) -> list[list[dict[str, Any]]]:
    """Serialize every pose in a result field without filling missing values."""
    poses: list[list[dict[str, Any]]] = []
    for pose in getattr(result, attribute, None) or []:
        poses.append(
            [
                {"index": index, **available_landmark_data(landmark)}
                for index, landmark in enumerate(pose)
            ]
        )
    return poses
