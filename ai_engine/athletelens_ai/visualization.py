"""Reusable MediaPipe pose and normalized-point drawing helpers."""

from __future__ import annotations

from typing import Any


POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


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
        if hasattr(connection, "start") and hasattr(connection, "end"):
            start_index, end_index = int(connection.start), int(connection.end)
        else:
            start_index, end_index = int(connection[0]), int(connection[1])
        if start_index >= len(points) or end_index >= len(points):
            continue
        start, end = points[start_index], points[end_index]
        if start is not None and end is not None:
            cv2.line(frame, pixel(start), pixel(end), (45, 220, 70), 2, cv2.LINE_AA)

    for point in points:
        if point is not None:
            cv2.circle(frame, pixel(point), 3, (20, 235, 250), -1, cv2.LINE_AA)


def draw_assessment_overlay(
    frame: Any,
    landmark_data: list[dict[str, Any]],
    annotation_lines: list[str],
    cv2: Any,
) -> None:
    """Draw observed pose points and supplied, response-backed labels."""
    indexed: list[tuple[float, float] | None] = [None] * 33
    for item in landmark_data:
        try:
            index = int(item["index"])
            x, y = float(item["x"]), float(item["y"])
            if 0 <= index < len(indexed):
                indexed[index] = (x, y)
        except (KeyError, TypeError, ValueError):
            continue
    draw_normalized_points(frame, indexed, list(POSE_CONNECTIONS), cv2)

    if not annotation_lines:
        return
    x, y = 12, 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, line_gap = 0.48, 1, 23
    widest = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in annotation_lines)
    panel_height = line_gap * len(annotation_lines) + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 7, y - 14), (x + widest + 10, y + panel_height - 14), (8, 16, 28), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    for index, line in enumerate(annotation_lines):
        cv2.putText(frame, line, (x, y + index * line_gap), font, scale, (226, 240, 252), thickness, cv2.LINE_AA)
