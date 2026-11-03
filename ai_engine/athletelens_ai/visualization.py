"""Reusable MediaPipe pose and normalized-point drawing helpers."""

from __future__ import annotations

from typing import Any


def draw_pose_landmarks(
    frame: Any,
    pose_landmarks: Any,
    connections: Any,
    drawing_utils: Any,
    landmark_drawing_spec: Any,
    connection_drawing_spec: Any,
) -> None:
    """Draw a MediaPipe normalized landmark list using its drawing utility."""
    drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=pose_landmarks,
        connections=connections,
        landmark_drawing_spec=landmark_drawing_spec,
        connection_drawing_spec=connection_drawing_spec,
    )


def draw_normalized_points(
    frame: Any,
    points: list[tuple[float, float] | None],
    connections: list[Any],
    cv2: Any,
) -> None:
    """Draw normalized x/y points and connections with the phase visualizer style."""
    height, width = frame.shape[:2]

    def pixel(point: tuple[float, float]) -> tuple[int, int]:
        x = min(width - 1, max(0, int(round(point[0] * (width - 1)))))
        y = min(height - 1, max(0, int(round(point[1] * (height - 1)))))
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
