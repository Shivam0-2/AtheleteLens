"""Experimental real-video diagnostic for the existing vertical-jump assessment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from athletelens_ai.assessments.vertical_jump import VerticalJumpAssessment, VerticalJumpConfig
from jump_diagnostic_utils import (collect_assessment, ensure_fresh_outputs,
                                  render_video, unique_run_directory, verify_mp4, write_rows_csv,
                                  write_signal_plot)


AI_ENGINE_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = AI_ENGINE_DIR / "videos" / "Verticaljump.mp4"
DEFAULT_MODEL = AI_ENGINE_DIR / "models" / "pose_landmarker_lite.task"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an experimental vertical-jump video diagnostic.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path,
                        default=unique_run_directory(AI_ENGINE_DIR / "experimental_outputs", "vertical_jump_diagnostic"))
    return parser.parse_args(argv)


def rows_for_csv(result: Any, observations: list[Any], measurements: list[Any], events: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = []
    count = 0
    for source, observation, measurement in zip(result.summary["measurements"], observations, measurements):
        row = dict(source)
        count += sum(event.get("event_type") == "accepted" for event in row.get("events", []))
        row["repetition_count"] = count
        row["pose_detected"] = bool(observation.pose_detected)
        row["visibility_score"] = measurement.quality.get("visibility_score")
        row["missing_landmarks"] = measurement.quality.get("missing_landmarks", "")
        row["event_annotations"] = []
        rows.append(row)
    if rows:
        for event in events or ():
            annotations = []
            for label, key in (("takeoff_candidate", "takeoff_candidate_timestamp_ms"),
                               ("takeoff_confirmed", "takeoff_confirmation_timestamp_ms"),
                               ("landing_candidate", "landing_candidate_timestamp_ms"),
                               ("landing_confirmed", "landing_confirmation_timestamp_ms")):
                if event.get(key) is not None:
                    annotations.append({"type": label, "timestamp_ms": event[key]})
            annotations.append({"type": event.get("event_type"), "reason": event.get("reason"),
                                "attempt_number": event.get("attempt_number")})
            anchor = event.get("end_timestamp_ms")
            if anchor is None:
                anchor = event.get("landing_confirmation_timestamp_ms")
            if anchor is None:
                anchor = event.get("takeoff_confirmation_timestamp_ms") or event.get("start_timestamp_ms")
            if anchor is not None:
                index = min(range(len(rows)), key=lambda i: abs(int(rows[i]["timestamp_ms"]) - int(anchor)))
                rows[index]["event_annotations"].extend(annotations)
    return rows


def event_frame_index(rows: list[dict[str, Any]], timestamp_ms: int) -> int:
    """Return the nearest decoded frame for an event timestamp."""
    if not rows:
        return 0
    return min(range(len(rows)), key=lambda i: abs(int(rows[i]["timestamp_ms"]) - timestamp_ms))


def run_diagnostic(video_path: Path, model_path: Path, output_dir: Path) -> dict[str, Any]:
    video_path, model_path, output_dir = (p.expanduser().resolve() for p in (video_path, model_path, output_dir))
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model does not exist: {model_path}")
    outputs = {"video": output_dir / "vertical_jump_annotated.mp4",
               "csv": output_dir / "vertical_jump_measurements.csv",
               "json": output_dir / "vertical_jump_summary.json",
               "plot": output_dir / "vertical_jump_displacement_vs_time.png"}
    ensure_fresh_outputs(outputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = VerticalJumpConfig()
    assessment = VerticalJumpAssessment(config)
    run = collect_assessment(video_path, model_path, assessment)
    rows = rows_for_csv(run["result"], run["observations"], run["measurements"], list(run["result"].events))
    result, metadata, cv2, np, observations = (run[k] for k in ("result", "metadata", "cv2", "np", "observations"))
    accepted = [dict(e) for e in result.events if e.get("event_type") == "accepted"]
    rejected = [dict(e) for e in result.events if e.get("event_type") == "rejected"]

    marker_by_frame: dict[int, list[str]] = {}
    for event in accepted:
        number = event.get("attempt_number", "?")
        for label, key in (("TAKEOFF CONFIRMED", "takeoff_confirmation_timestamp_ms"),
                           ("LANDING CONFIRMED", "landing_confirmation_timestamp_ms")):
            if event.get(key) is not None:
                marker_by_frame.setdefault(event_frame_index(rows, int(event[key])), []).append(f"{label} #{number}")
        marker_by_frame.setdefault(event_frame_index(rows, int(event["landing_timestamp_ms"])), []).append(f"ACCEPTED #{number}")
    for event in rejected:
        marker_by_frame.setdefault(event_frame_index(rows, int(event["end_timestamp_ms"])), []).append(
            f"REJECTED: {event.get('reason', 'unknown')}"
        )

    baseline_y = next((float(row["foot_ground_y"]) + float(row["vertical_displacement"])
                       for row in rows if row.get("foot_ground_y") is not None and row.get("vertical_displacement") is not None), None)

    def overlay(frame: Any, row: dict[str, Any], index: int, cv: Any) -> None:
        height, width = frame.shape[:2]
        if baseline_y is not None:
            y = min(height - 1, max(0, round(baseline_y * (height - 1))))
            cv.line(frame, (0, y), (width - 1, y), (255, 180, 20), 2, cv.LINE_AA)
            cv.putText(frame, "Standing baseline", (8, max(18, y - 8)), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 180, 20), 2, cv.LINE_AA)
        displacement = row.get("vertical_displacement")
        hip_rise = row.get("hip_rise")
        lines = ["AthleteLens | EXPERIMENTAL / UNVALIDATED",
                 f"State: {row.get('state', 'n/a')} | Attempt count: {row.get('repetition_count', 0)}",
                 f"Foot rise: {displacement:.4f} normalized image units" if displacement is not None else "Foot rise: unavailable",
                 f"Hip rise: {hip_rise:.4f} normalized image units" if hip_rise is not None else "Hip rise: unavailable",
                 "Tracking: valid" if row.get("measurement_valid") else "Tracking: missing / invalid"]
        panel = frame.copy()
        cv.rectangle(panel, (6, 6), (min(width-6, 560), 142), (25, 28, 35), -1)
        cv.addWeighted(panel, .78, frame, .22, 0, frame)
        for line_no, text in enumerate(lines):
            cv.putText(frame, text, (15, 27 + line_no*23), cv.FONT_HERSHEY_SIMPLEX, .48,
                       (245,245,245) if line_no != 1 else (255,220,120), 2, cv.LINE_AA)
        labels = marker_by_frame.get(index, [])
        if labels:
            cv.rectangle(frame, (width//2-180, height//2-36), (width//2+180, height//2+24), (25,130,35), -1)
            cv.putText(frame, " | ".join(labels)[:50], (max(10,width//2-165), height//2+2),
                       cv.FONT_HERSHEY_DUPLEX, .58, (255,255,255), 2, cv.LINE_AA)

    write_rows_csv(outputs["csv"], rows)
    rendered = render_video(video_path, outputs["video"], rows, observations, metadata, cv2,
                            run["mp"].tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS, overlay)
    write_signal_plot(outputs["plot"], rows, cv2, np,
                      "Vertical jump | vertical displacement vs time (experimental)",
                      [("Foot", "vertical_displacement", (185,75,35)), ("Hip", "hip_rise", (35,130,205))],
                      "Normalized image displacement")
    playback = verify_mp4(outputs["video"], rendered, metadata, cv2)
    pose_count = sum(bool(o.pose_detected) for o in observations)
    valid_count = sum(bool(m.quality.get("measurement_valid")) for m in run["measurements"])
    summary = {"status": "Experimental diagnostic; not validated against measured jump heights.",
               "input_video": str(video_path), "outputs": {k: str(v) for k,v in outputs.items()},
               "video": {"fps": metadata.fps, "width": metadata.width, "height": metadata.height,
                         "frames_processed": len(observations), "frames_rendered": rendered, **playback},
               "detector_configuration": dict(result.configuration),
               "detected_attempts": len(accepted),
               "accepted_attempts": accepted, "rejected_attempts": rejected,
               "quality": {"frames_with_pose": pose_count, "pose_detection_percentage": round(100*pose_count/len(observations),2),
                           "frames_with_valid_measurement": valid_count,
                           "measurement_coverage_percentage": round(100*valid_count/len(observations),2),
                           "mean_visibility": result.summary.get("mean_visibility"),
                           "missing_measurement_reasons": dict(Counter(str(m.quality.get("missing_landmarks", "")) for m in run["measurements"] if not m.quality.get("measurement_valid")))},
               "height_method": result.summary["height_method"], "height_assumption": result.summary["height_assumption"],
               "height_output_field": result.summary["height_output_field"],
               "height_definition": result.summary["height_definition"],
               "contact_detection_note": result.summary["contact_detection_note"],
               "notice": "No measurements or events were fabricated. Flight-time height assumes equal takeoff and landing height; synthetic or diagnostic outputs are not real-video accuracy validation."}
    outputs["json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = run_diagnostic(args.video, args.model, args.output_dir)
    except Exception as exc:
        print(f"Error: vertical-jump diagnostic failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(f"Experimental vertical-jump attempts accepted: {len(summary['accepted_attempts'])}")
    for name, path in summary["outputs"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
