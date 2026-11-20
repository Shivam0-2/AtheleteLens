"""Safe local annotated-video generation and retrieval helpers."""

from __future__ import annotations

import math
import os
import uuid
from pathlib import Path
from typing import Any

from athletelens_ai.video import iter_video_frames, open_video
from athletelens_ai.visualization import draw_assessment_overlay


class AnnotatedVideoError(RuntimeError):
    """An annotation could not be written or verified as a playable MP4."""


def create_annotated_video(
    source_path: Path,
    output_dir: Path,
    pose_frames: list[list[dict[str, Any]]],
    annotation_lines: list[list[str]],
    *,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
) -> tuple[str, str]:
    """Write one MP4 from saved pose observations without repeating inference."""
    try:
        import cv2
    except ImportError as exc:
        raise AnnotatedVideoError("OpenCV is unavailable for video export") from exc

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    video_id = uuid.uuid4().hex
    final_path = root / f"{video_id}.mp4"
    temporary_path = root / f".{video_id}.partial.mp4"
    capture = writer = None
    decoded_count = 0
    try:
        capture, metadata = open_video(source_path, cv2)
        if (metadata.width, metadata.height) != (expected_width, expected_height):
            raise AnnotatedVideoError("Source video dimensions changed during processing")
        if not math.isclose(metadata.fps, expected_fps, rel_tol=0.01, abs_tol=0.05):
            raise AnnotatedVideoError("Source video FPS changed during processing")
        writer = cv2.VideoWriter(
            str(temporary_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            raise AnnotatedVideoError("OpenCV could not initialize the MP4 writer")
        for frame_number, frame in iter_video_frames(capture):
            poses = pose_frames[frame_number] if frame_number < len(pose_frames) else []
            lines = annotation_lines[frame_number] if frame_number < len(annotation_lines) else []
            if not poses and "Pose not detected" not in lines:
                lines = [*lines, "Pose not detected"]
            draw_assessment_overlay(frame, poses, lines, cv2)
            writer.write(frame)
            decoded_count += 1
        if decoded_count == 0 or decoded_count != len(pose_frames):
            raise AnnotatedVideoError("The source frame count changed during annotation")
    except AnnotatedVideoError:
        raise
    except Exception as exc:
        raise AnnotatedVideoError(f"Could not annotate source video ({type(exc).__name__})") from exc
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()

    verify = None
    try:
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise AnnotatedVideoError("Annotated MP4 was not created")
        verify = cv2.VideoCapture(str(temporary_path))
        if not verify.isOpened():
            raise AnnotatedVideoError("Annotated MP4 cannot be opened")
        verified_count = 0
        while True:
            ok, frame = verify.read()
            if not ok:
                break
            if frame.shape[1] != expected_width or frame.shape[0] != expected_height:
                raise AnnotatedVideoError("Annotated MP4 dimensions do not match the source")
            verified_count += 1
        if verified_count != decoded_count:
            raise AnnotatedVideoError(
                f"Annotated MP4 decoded {verified_count} of {decoded_count} frames"
            )
        verified_fps = float(verify.get(cv2.CAP_PROP_FPS))
        if not math.isclose(verified_fps, expected_fps, rel_tol=0.02, abs_tol=0.1):
            raise AnnotatedVideoError("Annotated MP4 FPS does not match the source")
        verify.release()
        verify = None
        os.replace(temporary_path, final_path)
    except AnnotatedVideoError:
        raise
    except Exception as exc:
        raise AnnotatedVideoError(f"Annotated MP4 verification failed ({type(exc).__name__})") from exc
    finally:
        if verify is not None:
            verify.release()
        temporary_path.unlink(missing_ok=True)

    return video_id, f"/api/v1/assessments/annotated-videos/{video_id}"


def build_annotation_lines(
    exercise_type: str,
    frame_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    timestamps_ms: list[int],
) -> list[list[str]]:
    """Create per-frame labels only from measurements and events in the result."""
    lines: list[list[str]] = []
    frame_by_timestamp = {timestamp: index for index, timestamp in enumerate(timestamps_ms)}
    markers: dict[int, list[str]] = {}
    accepted_number = 0
    rejected_number = 0
    for event in events:
        accepted = event.get("event_type") in {"accepted", "completed"}
        timestamp = event.get("end_timestamp_ms")
        if timestamp is None:
            timestamp = event.get("landing_confirmation_timestamp_ms", event.get("landing_timestamp_ms"))
        if not isinstance(timestamp, (int, float)) or not timestamps_ms:
            continue
        index = min(range(len(timestamps_ms)), key=lambda i: abs(timestamps_ms[i] - int(timestamp)))
        if accepted:
            accepted_number += 1
            number = event.get("repetition_number", event.get("attempt_number", accepted_number))
            value = None
            if exercise_type == "vertical_jump" and isinstance(event.get("flight_time_height_cm"), (int, float)):
                value = f" · {event['flight_time_height_cm']:.2f} cm flight-time estimate"
            elif exercise_type == "standing_broad_jump":
                if event.get("physical_distance_available") is True and isinstance(event.get("distance_cm"), (int, float)):
                    value = f" · {event['distance_cm']:.1f} cm calibrated"
                elif isinstance(event.get("horizontal_displacement_px"), (int, float)):
                    value = f" · {event['horizontal_displacement_px']:.1f} px"
            markers.setdefault(index, []).append(f"Accepted {'repetition' if exercise_type in {'push_up', 'curl_up'} else 'attempt'} {number}{value or ''}")
        else:
            rejected_number += 1
            reason = str(event.get("reason") or "rejected")
            markers.setdefault(index, []).append(f"Rejected event {rejected_number}: {reason}")

    by_frame = {int(row["frame_number"]): row for row in frame_rows if isinstance(row.get("frame_number"), (int, float))}
    for frame_number, timestamp in enumerate(timestamps_ms):
        row = by_frame.get(frame_number, {})
        state = row.get("movement_state", row.get("state"))
        if exercise_type == "push_up":
            angle = row.get("angle_deg")
            metric = f"Elbow angle: {angle:.1f} deg" if isinstance(angle, (int, float)) else "Elbow angle: unavailable"
        elif exercise_type == "curl_up":
            angle = row.get("included_angle_deg")
            metric = f"Shoulder-hip-knee angle: {angle:.1f} deg" if isinstance(angle, (int, float)) else "Shoulder-hip-knee angle: unavailable"
        elif exercise_type == "vertical_jump":
            displacement = row.get("vertical_displacement")
            metric = f"Foot displacement: {displacement:.3f} normalized image units" if isinstance(displacement, (int, float)) else "Foot displacement: unavailable"
        else:
            displacement = row.get("horizontal_displacement_px")
            metric = f"Heel displacement: {displacement:.1f} px" if isinstance(displacement, (int, float)) else "Heel displacement: unavailable"
            physical_cm = row.get("physical_distance_cm")
            if isinstance(physical_cm, (int, float)) and row.get("physical_distance_available") is True:
                metric += f" · {physical_cm:.1f} cm calibrated"
        frame_lines = [exercise_type.replace("_", " ").title(), metric]
        if state is not None:
            frame_lines.append(f"Phase: {state}")
        frame_lines.extend(markers.get(frame_number, []))
        lines.append(frame_lines)
    return lines
