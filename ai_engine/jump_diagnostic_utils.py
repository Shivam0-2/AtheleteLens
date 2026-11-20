"""Shared export helpers for the experimental jump diagnostic CLIs."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from athletelens_ai.pipeline import PoseVideoPipeline
from athletelens_ai.video import iter_video_frames, open_video
from athletelens_ai.visualization import draw_normalized_points


def unique_run_directory(root: Path, assessment: str) -> Path:
    """Choose a fresh run directory without replacing earlier experiment files."""
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = root / assessment / stamp
    suffix = 2
    while candidate.exists():
        candidate = root / assessment / f"{stamp}_{suffix}"
        suffix += 1
    return candidate


def ensure_fresh_outputs(paths: dict[str, Path]) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing diagnostic outputs: "
            + ", ".join(str(path) for path in existing)
            + ". Choose a fresh --output-dir."
        )


def collect_assessment(video_path: Path, model_path: Path, assessment: Any) -> dict[str, Any]:
    """Run one shared pose pass and then the assessment's temporal analysis."""
    observations = []
    measurements = []
    with PoseVideoPipeline(video_path, model_path) as pipeline:
        metadata = pipeline.metadata
        if metadata is None:
            raise RuntimeError("Pose pipeline did not provide video metadata")
        for observation in pipeline:
            observations.append(observation)
            measurements.append(assessment.measure_frame(observation))
        cv2, mp, np = pipeline.cv2, pipeline.mp, pipeline.np
    if not observations:
        raise ValueError(f"Video contains no readable frames: {video_path}")
    result = assessment.analyze(measurements, metadata)
    rows = list(result.summary.get("measurements", ()))
    if len(rows) != len(observations):
        raise RuntimeError("Assessment measurement rows do not match processed video frames")
    return {"observations": observations, "measurements": measurements, "metadata": metadata,
            "result": result, "rows": rows, "cv2": cv2, "mp": mp, "np": np}


def normalized_pose_points(observation: Any) -> list[tuple[float, float] | None]:
    """Return normalized drawable pose points while preserving absent points."""
    if not observation.pose_landmark_data:
        return []
    points = []
    for landmark in observation.pose_landmark_data[0]:
        try:
            x, y = float(landmark["x"]), float(landmark["y"])
        except (KeyError, TypeError, ValueError):
            points.append(None)
            continue
        points.append((x, y) if math.isfinite(x) and math.isfinite(y) else None)
    return points


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-frame records; absent values remain blank rather than filled."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def write_signal_plot(path: Path, rows: list[dict[str, Any]], cv2: Any, np: Any,
                      title: str, series: list[tuple[str, str, tuple[int, int, int]]],
                      y_label: str) -> None:
    """Plot numeric per-frame series; None values break the line into segments."""
    width, height = 1200, 720
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 100, width - 45, 65, height - 90
    valid_values = []
    for _, key, _ in series:
        valid_values.extend(float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key])))
    low, high = (min(valid_values), max(valid_values)) if valid_values else (0.0, 1.0)
    if math.isclose(low, high):
        low -= 0.5
        high += 0.5
    padding = (high - low) * 0.08
    low, high = low - padding, high + padding
    max_time = max((int(row["timestamp_ms"]) for row in rows), default=1000) / 1000.0
    max_time = max(max_time, 1.0)
    cv2.putText(canvas, title, (left, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (25, 25, 25), 2, cv2.LINE_AA)
    for fraction in range(5):
        yval = low + (high - low) * fraction / 4
        y = int(bottom - (yval - low) / (high - low) * (bottom - top))
        cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)
        cv2.putText(canvas, f"{yval:.3f}", (left - 78, y + 4), cv2.FONT_HERSHEY_SIMPLEX, .38, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (left, bottom), (40, 40, 40), 2)
    cv2.line(canvas, (left, bottom), (right, bottom), (40, 40, 40), 2)
    cv2.putText(canvas, y_label, (8, (top + bottom) // 2), cv2.FONT_HERSHEY_SIMPLEX, .42, (35, 35, 35), 1, cv2.LINE_AA)
    for label, key, color in series:
        previous = None
        for row in rows:
            value = row.get(key)
            if value is None or not math.isfinite(float(value)):
                previous = None
                continue
            x = int(left + (int(row["timestamp_ms"]) / 1000.0 / max_time) * (right - left))
            y = int(bottom - (float(value) - low) / (high - low) * (bottom - top))
            current = (x, y)
            if previous is not None:
                cv2.line(canvas, previous, current, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, current, 2, color, -1, cv2.LINE_AA)
            previous = current
        cv2.putText(canvas, label, (left + 12 + 280 * series.index((label, key, color)), top - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write measurement graph: {path}")


def render_video(
    source: Path,
    destination: Path,
    rows: list[dict[str, Any]],
    observations: list[Any],
    metadata: Any,
    cv2: Any,
    connections: Any,
    overlay: Callable[[Any, dict[str, Any], int, Any], None],
    *,
    calibration: Any = None,
    reference_points: tuple[tuple[float, float], tuple[float, float]] | None = None,
    reference_label: str | None = None,
) -> int:
    """Re-read source pixels and draw the already computed pose/assessment rows."""
    capture, render_metadata = open_video(source, cv2, "Could not reopen video for annotated render")
    writer = None
    try:
        if (render_metadata.fps, render_metadata.width, render_metadata.height) != (
            metadata.fps, metadata.width, metadata.height
        ):
            raise RuntimeError("Video metadata changed between inference and rendering")
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                                 metadata.fps, (metadata.width, metadata.height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create annotated MP4: {destination}")
        rendered = 0
        for frame_index, frame in iter_video_frames(capture):
            if frame_index >= len(rows) or frame_index >= len(observations):
                raise RuntimeError("Decoded frame count changed during render")
            draw_normalized_points(frame, normalized_pose_points(observations[frame_index]), connections, cv2)
            if calibration is not None or reference_points is not None:
                source_points = (calibration.endpoint_a_px, calibration.endpoint_b_px) if calibration is not None else reference_points
                a = tuple(int(round(v)) for v in source_points[0])
                b = tuple(int(round(v)) for v in source_points[1])
                cv2.line(frame, a, b, (0, 180, 255), 3, cv2.LINE_AA)
                cv2.circle(frame, a, 7, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, b, 7, (0, 255, 255), -1, cv2.LINE_AA)
                label = reference_label or (f"Ground reference: {calibration.known_length_m:g} m" if calibration is not None else "Ground reference: invalid calibration")
                cv2.putText(frame, label,
                            (min(a[0], b[0]), max(22, min(a[1], b[1]) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 180, 255), 2, cv2.LINE_AA)
            overlay(frame, rows[frame_index], frame_index, cv2)
            writer.write(frame)
            rendered += 1
        if rendered != len(rows):
            raise RuntimeError(f"Rendered {rendered} frames; expected {len(rows)}")
        return rendered
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def verify_mp4(path: Path, expected_frames: int, metadata: Any, cv2: Any) -> dict[str, Any]:
    """Decode the entire generated MP4 and verify frame count and dimensions."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Generated MP4 is not readable: {path}")
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != metadata.width or frame.shape[0] != metadata.height:
                raise RuntimeError("Generated MP4 frame dimensions do not match source")
            count += 1
        if count != expected_frames:
            raise RuntimeError(f"Generated MP4 decoded {count} frames; expected {expected_frames}")
        return {"decoded_frames": count,
                "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                "width": metadata.width, "height": metadata.height,
                "playable_decode_check": "passed"}
    finally:
        capture.release()
