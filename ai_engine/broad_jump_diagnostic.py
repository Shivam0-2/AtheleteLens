"""Experimental real-video standing broad-jump diagnostic and calibration UI."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from athletelens_ai.assessments.broad_jump import BroadJumpAssessment, BroadJumpConfig
from athletelens_ai.assessments.calibration import CalibrationError, GroundReferenceCalibration
from jump_diagnostic_utils import (collect_assessment, ensure_fresh_outputs, render_video,
                                  unique_run_directory, verify_mp4, write_rows_csv,
                                  write_signal_plot)


AI_ENGINE_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = AI_ENGINE_DIR / "videos" / "broadjump.mp4"
DEFAULT_MODEL = AI_ENGINE_DIR / "models" / "pose_landmarker_lite.task"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the experimental standing broad-jump diagnostic.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path,
                        default=unique_run_directory(AI_ENGINE_DIR / "experimental_outputs", "broad_jump_diagnostic"))
    parser.add_argument("--reference-points", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Use supplied pixel endpoints instead of mouse calibration.")
    parser.add_argument("--reference-length-m", type=float,
                        help="Known physical distance between the two reference endpoints.")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Run non-interactively and report pixel displacement only.")
    return parser.parse_args(argv)


def calibration_from_input(points: tuple[float, float, float, float] | list[float],
                           known_length_m: float, width: int, height: int
                           ) -> tuple[GroundReferenceCalibration | None, dict[str, Any]]:
    """Validate image endpoints and length using the shared calibration class."""
    reference = GroundReferenceCalibration((points[0], points[1]), (points[2], points[3]), known_length_m)
    try:
        resolved = reference.calibrate(width, height)
    except (CalibrationError, TypeError, ValueError) as exc:
        return None, {"status": "invalid", "error": str(exc), "endpoints_px": [list(reference.endpoint_a_px), list(reference.endpoint_b_px)],
                      "known_length_m": known_length_m, "physical_distance_available": False}
    return reference, {"status": "valid", **resolved.to_dict(), "physical_distance_available": True}


def interactive_reference(video_path: Path, cv2: Any) -> tuple[list[float], float] | None:
    """Show a middle frame, collect two mouse clicks, then ask for real length."""
    from athletelens_ai.video import open_video
    capture, metadata = open_video(video_path, cv2, "Could not open video for calibration preview")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
        ok, original = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, original = capture.read()
        if not ok:
            raise ValueError("Could not read a representative calibration frame")
        points: list[tuple[int, int]] = []
        window = "AthleteLens broad-jump ground calibration"

        def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)
        try:
            while True:
                display = original.copy()
                cv2.putText(display, "Click reference endpoints; Enter accepts, Esc skips calibration",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 2, cv2.LINE_AA)
                for point in points:
                    cv2.circle(display, point, 8, (0, 255, 255), -1, cv2.LINE_AA)
                if len(points) == 2:
                    cv2.line(display, points[0], points[1], (0, 180, 255), 3, cv2.LINE_AA)
                cv2.imshow(window, display)
                key = cv2.waitKey(20) & 0xFF
                if key == 27:
                    return None
                if key in (10, 13) and len(points) == 2:
                    break
        finally:
            cv2.destroyWindow(window)
        try:
            length = float(input("Known distance between clicked endpoints (meters): ").strip())
        except (ValueError, EOFError):
            return None
        return [float(points[0][0]), float(points[0][1]), float(points[1][0]), float(points[1][1])], length
    finally:
        capture.release()


def rows_for_csv(result: Any, observations: list[Any], measurements: list[Any],
                 scale: float | None, *, calibration_status: str | None = None,
                 calibration_explanation: str | None = None) -> list[dict[str, Any]]:
    """Attach tracking metadata without changing assessment pixel units."""
    rows = []
    accepted_count = 0
    calibration = result.summary.get("calibration", {})
    for source, observation, measurement in zip(result.summary["measurements"], observations, measurements):
        row = dict(source)
        row["pose_detected"] = bool(observation.pose_detected)
        row["visibility_score"] = measurement.quality.get("visibility_score")
        row["missing_landmarks"] = measurement.quality.get("missing_landmarks", "")
        row["selected_foot_landmarks"] = measurement.quality.get("selected_foot_landmarks", "")
        row.setdefault("physical_distance_m", None)
        row.setdefault("physical_distance_cm", None)
        row["physical_distance_available"] = bool(scale is not None and row.get("physical_distance_available"))
        if any(event.get("event_type") == "accepted" for event in row.get("events", [])):
            accepted_count += 1
        row["accepted_movement_attempt_count"] = accepted_count
        row["calibration_status"] = calibration_status or calibration.get("status", "unavailable")
        row["calibration_explanation"] = calibration_explanation or calibration.get("explanation", "No calibration metadata supplied.")
        rows.append(row)
    return rows


def estimate_start_heel(rows: list[dict[str, Any]], confirmation_frames: int) -> tuple[float, float] | None:
    """Recover the assessment's initial stable heel center for a visual marker."""
    baseline = []
    for row in rows:
        if row.get("heel_center_x_px") is not None and row.get("heel_ground_y") is not None:
            baseline.append((float(row["heel_center_x_px"]), float(row["heel_ground_y"])))
        if len(baseline) >= confirmation_frames:
            return (sum(p[0] for p in baseline) / len(baseline), sum(p[1] for p in baseline) / len(baseline))
        if row.get("movement_state") not in {"SEEK_STARTING_POSITION", "STANDING"}:
            break
    return None


def run_diagnostic(video_path: Path, model_path: Path, output_dir: Path,
                   calibration_input: tuple[list[float], float] | None = None,
                   calibration_status: dict[str, Any] | None = None) -> dict[str, Any]:
    video_path, model_path, output_dir = (p.expanduser().resolve() for p in (video_path, model_path, output_dir))
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model does not exist: {model_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model does not exist: {model_path}")
    outputs = {"video": output_dir / "broad_jump_annotated.mp4",
               "csv": output_dir / "broad_jump_measurements.csv",
               "json": output_dir / "broad_jump_summary.json",
               "plot": output_dir / "broad_jump_displacement_vs_time.png"}
    ensure_fresh_outputs(outputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2 = None
    reference = None
    if calibration_input is not None:
        import cv2 as cv2_module
        cv2 = cv2_module
        points, known_length = calibration_input
        # The preview-frame size is checked against source dimensions again by assessment.
        from athletelens_ai.video import open_video
        preview_capture, preview_metadata = open_video(video_path, cv2, "Could not open calibration input video")
        preview_capture.release()
        reference, detected_status = calibration_from_input(points, known_length, preview_metadata.width, preview_metadata.height)
        calibration_status = detected_status
    else:
        import cv2 as cv2_module
        cv2 = cv2_module
        calibration_status = calibration_status or {"status": "unavailable", "physical_distance_available": False,
                                                     "reason": "No calibration was supplied; output is pixels only."}
    assessment = BroadJumpAssessment(BroadJumpConfig(ground_reference=reference))
    run = collect_assessment(video_path, model_path, assessment)
    result, metadata, np, observations = (run[k] for k in ("result", "metadata", "np", "observations"))
    scale = result.summary["calibration"]["pixels_per_meter"]
    reported_calibration_status = result.summary["calibration"]["status"]
    reported_calibration_explanation = result.summary["calibration"]["explanation"]
    if scale is None and calibration_status and calibration_status.get("status") == "invalid":
        reported_calibration_status = "invalid"
        reported_calibration_explanation = calibration_status.get("error") or "The supplied calibration is invalid."
    rows = rows_for_csv(result, observations, run["measurements"], scale,
                        calibration_status=reported_calibration_status,
                        calibration_explanation=reported_calibration_explanation)
    accepted = [dict(event) for event in result.events if event.get("event_type") == "accepted"]
    rejected = [dict(event) for event in result.events if event.get("event_type") == "rejected"]
    if reported_calibration_status == "invalid":
        explanation = reported_calibration_explanation
        for event in accepted + rejected:
            event["calibration_status"] = "invalid"
            event["calibration_explanation"] = explanation
    start_heel = estimate_start_heel(rows, int(result.configuration["baseline_confirmation_frames"]))
    timestamps = [int(row["timestamp_ms"]) for row in rows]
    marker_by_frame: dict[int, list[str]] = {}
    for event in accepted:
        number = event.get("attempt_number", "?")
        marker_by_frame.setdefault(min(range(len(timestamps)), key=lambda i: abs(timestamps[i]-int(event["takeoff_timestamp_ms"]))), []).append(f"TAKEOFF #{number}")
        marker_by_frame.setdefault(min(range(len(timestamps)), key=lambda i: abs(timestamps[i]-int(event["landing_timestamp_ms"]))), []).append(f"LANDING / ACCEPTED #{number}")
    for event in rejected:
        ts = int(event["end_timestamp_ms"])
        marker_by_frame.setdefault(min(range(len(timestamps)), key=lambda i: abs(timestamps[i]-ts)), []).append(f"REJECTED: {event.get('reason')}")

    def overlay(frame: Any, row: dict[str, Any], index: int, cv: Any) -> None:
        height, width = frame.shape[:2]
        if start_heel is not None:
            start_px = (int(round(start_heel[0])), int(round(start_heel[1] * (height - 1))))
            cv.circle(frame, start_px, 8, (255, 190, 0), 2, cv.LINE_AA)
            cv.putText(frame, "Starting heel", (start_px[0] + 8, start_px[1] - 8), cv.FONT_HERSHEY_SIMPLEX, .48, (255,190,0), 2, cv.LINE_AA)
        labels = marker_by_frame.get(index, [])
        if any(label.startswith(("LANDING", "REJECTED")) for label in labels) and row.get("heel_center_x_px") is not None:
            heel = (int(round(float(row["heel_center_x_px"]))), int(round(float(row["heel_ground_y"]) * (height - 1))))
            cv.circle(frame, heel, 10, (20, 30, 240), 3, cv.LINE_AA)
            cv.putText(frame, "Landing heel", (heel[0] + 8, heel[1] - 8), cv.FONT_HERSHEY_SIMPLEX, .48, (20,30,240), 2, cv.LINE_AA)
        dx = row.get("horizontal_displacement_px")
        meters = row.get("physical_distance_m")
        calibration_state = row.get("calibration_status", "unavailable")
        distance_text = (f"Physical distance: {float(meters):.3f} m" if meters is not None else
                         f"Physical distance: unavailable ({calibration_state} calibration)")
        lines = ["AthleteLens | EXPERIMENTAL / UNVALIDATED",
                 f"State: {row.get('movement_state')} | Accepted attempts: {row.get('accepted_movement_attempt_count', 0)}",
                 f"Heel displacement: {float(dx):.1f} px" if dx is not None else "Heel displacement: unavailable",
                 distance_text,
                 f"Tracking visibility: {row.get('visibility_score'):.2f}" if row.get("visibility_score") is not None else "Tracking visibility: unavailable"]
        panel = frame.copy()
        cv.rectangle(panel, (6,6), (min(width-6,570),142), (25,28,35), -1)
        cv.addWeighted(panel,.78,frame,.22,0,frame)
        for n, text in enumerate(lines):
            cv.putText(frame,text,(15,27+n*23),cv.FONT_HERSHEY_SIMPLEX,.48,(245,245,245) if n!=1 else (255,220,120),2,cv.LINE_AA)
        if labels:
            cv.rectangle(frame,(width//2-200,height//2-34),(width//2+200,height//2+24),(25,130,35),-1)
            cv.putText(frame," | ".join(labels)[:52],(max(8,width//2-185),height//2+2),cv.FONT_HERSHEY_DUPLEX,.52,(255,255,255),2,cv.LINE_AA)

    write_rows_csv(outputs["csv"], rows)
    rendered = render_video(video_path, outputs["video"], rows, observations, metadata, cv2,
                            run["mp"].tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS, overlay,
                            calibration=reference,
                            reference_points=((calibration_input[0][0], calibration_input[0][1]),
                                              (calibration_input[0][2], calibration_input[0][3])) if calibration_input else None,
                            reference_label=(f"Ground reference: {calibration_input[1]:g} m" if reference is not None else
                                             "Ground reference: INVALID" if calibration_input else None))
    write_signal_plot(outputs["plot"], rows, cv2, np,
                      "Standing broad jump | heel displacement vs time (experimental)",
                      [("Heel displacement (pixels)", "horizontal_displacement_px", (185,75,35))],
                      "Horizontal displacement (pixels)")
    playback = verify_mp4(outputs["video"], rendered, metadata, cv2)
    pose_count = sum(bool(o.pose_detected) for o in observations)
    valid_count = sum(bool(m.quality.get("measurement_valid")) for m in run["measurements"])
    px_values = [event.get("horizontal_displacement_px") for event in accepted + rejected if event.get("horizontal_displacement_px") is not None]
    summary = {"status": "Experimental diagnostic; not validated against manually measured jump distances.",
               "input_video": str(video_path), "outputs": {key: str(value) for key,value in outputs.items()},
               "video": {"fps": metadata.fps, "width": metadata.width, "height": metadata.height,
                         "frames_processed": len(observations), "frames_rendered": rendered, **playback},
               "detector_configuration": dict(result.configuration),
               "calibration": {**(calibration_status or {}), **result.summary["calibration"],
                               "status": reported_calibration_status,
                               "error": result.summary["calibration"].get("error") or (calibration_status or {}).get("error"),
                               "explanation": reported_calibration_explanation or (calibration_status or {}).get("reason"),
                               "physical_distance_available": result.summary["physical_distance_available"]},
               "accepted_movement_attempt_count": len(accepted),
               "physical_distance_attempt_count": sum(bool(event.get("physical_distance_available")) for event in accepted),
               "accepted_attempts": accepted,
               "attempts_with_physical_distance": [event for event in accepted if event.get("physical_distance_available")],
               "rejected_attempts": rejected,
               "pixel_displacements": px_values,
               "physical_distances_m": [event["distance_m"] for event in accepted if event.get("distance_m") is not None],
               "quality": {"frames_with_pose": pose_count, "pose_detection_percentage": round(100*pose_count/len(observations),2),
                           "frames_with_valid_foot_measurement": valid_count,
                           "measurement_coverage_percentage": round(100*valid_count/len(observations),2),
                           "mean_visibility": result.summary.get("mean_visibility"),
                           "missing_measurement_reasons": dict(Counter(str(m.quality.get("missing_landmarks", "")) for m in run["measurements"] if not m.quality.get("measurement_valid")))},
               "perspective_limitation": "A scalar scale applies locally near the flat reference and at a comparable camera depth. No perspective correction is performed.",
               "coordinate_handling": "Horizontal heel positions and displacement are converted from normalized MediaPipe x to image pixels using the actual video width before thresholds and calibration are applied.",
               "measurement_units": result.summary["measurement_units"],
               "notice": "Movement acceptance is independent of physical calibration. Uncalibrated attempts retain pixel displacement and null physical distances; incomplete or poorly tracked cycles remain rejected."}
    outputs["json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"Error: video does not exist: {video}", file=sys.stderr)
        return 2
    if (args.reference_points is None) != (args.reference_length_m is None):
        print("Error: --reference-points and --reference-length-m must be supplied together.", file=sys.stderr)
        return 2
    if args.no_calibration and args.reference_points is not None:
        print("Error: --no-calibration cannot be combined with reference arguments.", file=sys.stderr)
        return 2
    try:
        calibration_input = None
        status = None
        if args.reference_points is not None:
            calibration_input = (args.reference_points, args.reference_length_m)
        elif not args.no_calibration:
            import cv2
            calibration_input = interactive_reference(video, cv2)
            if calibration_input is None:
                status = {"status": "unavailable", "physical_distance_available": False,
                          "reason": "Interactive calibration skipped or input length unavailable."}
        summary = run_diagnostic(video, args.model, args.output_dir, calibration_input, status)
    except Exception as exc:
        print(f"Error: broad-jump diagnostic failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(f"Experimental movement attempts accepted: {summary['accepted_movement_attempt_count']}")
    print(f"Accepted attempts with physical distance: {summary['physical_distance_attempt_count']}")
    for name, path in summary["outputs"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
