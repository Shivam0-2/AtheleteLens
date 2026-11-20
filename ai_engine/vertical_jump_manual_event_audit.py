"""Prepare frame-level human review material from a saved jump diagnostic.

This tool does not rerun pose estimation or alter detector events. It extracts
the nearest encoded diagnostic frame for each saved event timestamp and adds
only audit labels for human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


AI_ENGINE_DIR = Path(__file__).resolve().parent
DEFAULT_DIAGNOSTIC = AI_ENGINE_DIR / "experimental_outputs" / "vertical_jump_diagnostic" / "run_event_confirmation_01"
DEFAULT_OUTPUT = AI_ENGINE_DIR / "experimental_outputs" / "vertical_jump_diagnostic" / "manual_event_audit" / "run_event_confirmation_01"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export labeled manual-review frames for saved vertical-jump events.")
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def nearest_row(rows: list[dict[str, Any]], timestamp_ms: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("Diagnostic CSV contains no frame rows")
    return min(rows, key=lambda row: abs(int(row["timestamp_ms"]) - timestamp_ms))


def _audit_frame(frame: Any, row: dict[str, Any], label: str, event_timestamp: int,
                 cv2: Any, detail: str) -> Any:
    """Annotate a diagnostic frame without changing any detector result."""
    height, width = frame.shape[:2]
    # Add a labeled horizontal guide at the same foot-height proxy used by the
    # detector. This is explicitly a pose proxy, not proof of ground contact.
    foot_y = row.get("foot_ground_y")
    if foot_y not in (None, ""):
        y = min(height - 1, max(0, round(float(foot_y) * (height - 1))))
        cv2.line(frame, (0, y), (width - 1, y), (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "lowest visible foot/ankle y proxy", (12, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 165, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (width, 66), (18, 22, 30), -1)
    cv2.putText(frame, f"MANUAL EVENT AUDIT | {label}", (14, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .62, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, f"frame {int(row['frame_number'])} | source timestamp {int(row['timestamp_ms'])} ms | event {event_timestamp} ms | {detail}",
                (14, 51), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 225, 255), 1, cv2.LINE_AA)
    return frame


def _sheet_cell(image: Any, title: str, cv2: Any, width: int = 560) -> Any:
    height = round(image.shape[0] * width / image.shape[1])
    thumb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cell = cv2.copyMakeBorder(thumb, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 30, 38))
    cv2.putText(cell, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 245, 245), 1, cv2.LINE_AA)
    return cell


def _stack_grid(cells: list[Any], columns: int, cv2: Any) -> Any:
    if not cells:
        raise ValueError("No event frames available to assemble")
    cell_h, cell_w = cells[0].shape[:2]
    rows = (len(cells) + columns - 1) // columns
    canvas = cv2.copyMakeBorder(
        cv2.resize(cells[0], (cell_w, cell_h)), 0, rows * cell_h - cell_h,
        0, columns * cell_w - cell_w, cv2.BORDER_CONSTANT, value=(245, 245, 245)
    )
    for index, cell in enumerate(cells):
        y, x = divmod(index, columns)
        canvas[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w] = cell
    return canvas


def prepare_audit(diagnostic_dir: Path, output_dir: Path) -> dict[str, Any]:
    import cv2

    diagnostic_dir = diagnostic_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output directory: {output_dir}")
    summary_path = diagnostic_dir / "vertical_jump_summary.json"
    csv_path = diagnostic_dir / "vertical_jump_measurements.csv"
    video_path = diagnostic_dir / "vertical_jump_annotated.mp4"
    for path in (summary_path, csv_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required diagnostic artifact is missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as stream:
        frame_rows = list(csv.DictReader(stream))
    if not frame_rows:
        raise ValueError("Diagnostic CSV contains no measurements")

    accepted = list(summary.get("accepted_attempts", []))
    rejected = list(summary.get("rejected_attempts", []))
    requested: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    manifest: list[dict[str, Any]] = []
    attempt_specs: list[list[dict[str, Any]]] = []
    for event in accepted:
        attempt_no = int(event["attempt_number"])
        specs = [
            {"kind": "takeoff_candidate", "timestamp_ms": int(event["takeoff_candidate_timestamp_ms"]),
             "label": "TAKEOFF CANDIDATE", "paired_timestamp_ms": int(event["takeoff_confirmation_timestamp_ms"])},
            {"kind": "takeoff_confirmed", "timestamp_ms": int(event["takeoff_confirmation_timestamp_ms"]),
             "label": "TAKEOFF CONFIRMED", "paired_timestamp_ms": int(event["takeoff_candidate_timestamp_ms"])},
            {"kind": "landing_candidate", "timestamp_ms": int(event["landing_candidate_timestamp_ms"]),
             "label": "LANDING CANDIDATE", "paired_timestamp_ms": int(event["landing_confirmation_timestamp_ms"])},
            {"kind": "landing_confirmed", "timestamp_ms": int(event["landing_confirmation_timestamp_ms"]),
             "label": "LANDING CONFIRMED", "paired_timestamp_ms": int(event["landing_candidate_timestamp_ms"])},
        ]
        for spec in specs:
            row = nearest_row(frame_rows, spec["timestamp_ms"])
            frame_no = int(row["frame_number"])
            spec.update({"attempt_number": attempt_no, "frame_number": frame_no,
                         "frame_timestamp_ms": int(row["timestamp_ms"]),
                         "confirmation_delta_ms": abs(spec["paired_timestamp_ms"] - spec["timestamp_ms"])})
            requested[frame_no] = (row, requested.get(frame_no, (row, []))[1] + [spec])
            manifest.append({"attempt_number": attempt_no, "event": spec["kind"],
                             "event_timestamp_ms": spec["timestamp_ms"], "frame_number": frame_no,
                             "frame_timestamp_ms": spec["frame_timestamp_ms"],
                             "paired_timestamp_ms": spec["paired_timestamp_ms"],
                             "candidate_confirmation_delta_ms": spec["confirmation_delta_ms"]})
        attempt_specs.append(specs)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not read annotated diagnostic video: {video_path}")
    extracted: dict[int, Any] = {}
    try:
        max_frame = max(requested, default=-1)
        for frame_no in range(max_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Diagnostic MP4 ended before requested frame {frame_no}")
            if frame_no in requested:
                extracted[frame_no] = frame.copy()
    finally:
        capture.release()

    output_dir.mkdir(parents=True)
    frames_dir = output_dir / "accepted_attempt_frames"
    frames_dir.mkdir()
    sheets_dir = output_dir / "attempt_contact_sheets"
    sheets_dir.mkdir()
    all_cells = []
    sheet_records = []
    for attempt_index, (event, specs) in enumerate(zip(accepted, attempt_specs), start=1):
        attempt_cells = []
        for spec in specs:
            row = nearest_row(frame_rows, spec["timestamp_ms"])
            frame = extracted[spec["frame_number"]].copy()
            delta = spec["confirmation_delta_ms"]
            direction = "candidate to confirmation" if spec["kind"].endswith("candidate") else "confirmation from candidate"
            detail = f"{direction}: {delta} ms"
            frame = _audit_frame(frame, row, f"ATTEMPT {attempt_index} | {spec['label']}", spec["timestamp_ms"], cv2, detail)
            safe_kind = spec["kind"]
            filename = f"attempt_{attempt_index:02d}_{safe_kind}_frame_{spec['frame_number']:04d}_t_{spec['timestamp_ms']}ms.png"
            image_path = frames_dir / filename
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Could not write audit frame: {image_path}")
            sheet_title = f"A{attempt_index} {spec['label']} | f{spec['frame_number']} t={spec['timestamp_ms']}ms | delta {delta}ms"
            cell = _sheet_cell(frame, sheet_title, cv2)
            attempt_cells.append(cell)
            all_cells.append(cell)
            spec["image_path"] = str(image_path)
        sheet = _stack_grid(attempt_cells, 2, cv2)
        sheet_path = sheets_dir / f"attempt_{attempt_index:02d}_four_event_frames.png"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise RuntimeError(f"Could not write contact sheet: {sheet_path}")
        sheet_records.append({"attempt_number": attempt_index, "contact_sheet": str(sheet_path), "frames": specs})

    overall_sheet = _stack_grid(all_cells, 4, cv2)
    overall_path = output_dir / "all_six_attempts_contact_sheet.png"
    if not cv2.imwrite(str(overall_path), overall_sheet):
        raise RuntimeError(f"Could not write all-attempt contact sheet: {overall_path}")
    with (output_dir / "accepted_event_frame_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["attempt_number", "event", "event_timestamp_ms", "frame_number", "frame_timestamp_ms",
                   "paired_timestamp_ms", "candidate_confirmation_delta_ms"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manifest)

    rejected_dir = output_dir / "rejected_incomplete_attempt"
    rejected_dir.mkdir()
    rejected_frames = []
    for index, event in enumerate(rejected, start=1):
        times = [("attempt_start", event.get("attempt_start_timestamp_ms")),
                 ("end_rejected", event.get("end_timestamp_ms"))]
        for kind, time_ms in times:
            if time_ms is None:
                continue
            row = nearest_row(frame_rows, int(time_ms))
            frame_no = int(row["frame_number"])
            if frame_no not in extracted:
                cap = cv2.VideoCapture(str(video_path))
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                    ok, frame = cap.read()
                finally:
                    cap.release()
                if not ok:
                    raise RuntimeError(f"Could not read rejected-attempt frame {frame_no}")
            else:
                frame = extracted[frame_no].copy()
            frame = _audit_frame(frame, row, f"REJECTED ATTEMPT {index} | {event['reason']} | {kind}",
                                 int(time_ms), cv2, "separate rejected-attempt evidence")
            image_path = rejected_dir / f"rejected_{index:02d}_{kind}_frame_{frame_no:04d}_t_{int(time_ms)}ms.png"
            cv2.imwrite(str(image_path), frame)
            rejected_frames.append({"kind": kind, "event_timestamp_ms": int(time_ms),
                                    "frame_number": frame_no, "frame_timestamp_ms": int(row["timestamp_ms"]),
                                    "image_path": str(image_path)})
    rejected_report = {"status": "Rejected attempt kept separate from accepted-attempt contact sheets.",
                       "diagnostic_summary": str(summary_path), "rejected_attempts": rejected,
                       "review_frames": rejected_frames,
                       "note": "Pose-based foot/ankle event timing is a visual proxy, not force-plate ground-contact measurement."}
    rejected_path = rejected_dir / "rejected_attempt_report.json"
    rejected_path.write_text(json.dumps(rejected_report, indent=2), encoding="utf-8")

    report = {"status": "Manual review preparation only; no events or measurements were changed.",
              "diagnostic_summary": str(summary_path), "annotated_video": str(video_path),
              "output_directory": str(output_dir), "accepted_attempt_count": len(accepted),
              "attempts": sheet_records, "all_attempts_contact_sheet": str(overall_path),
              "rejected_attempt_report": str(rejected_path),
              "contact_proxy_notice": "Candidate and confirmation markers come from pose-landmark thresholds. They are not equivalent to force-plate or contact-mat evidence."}
    (output_dir / "manual_event_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    try:
        report = prepare_audit(args.diagnostic_dir, args.output_dir)
    except Exception as exc:
        print(f"Error: manual event audit preparation failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(f"Accepted attempts prepared: {report['accepted_attempt_count']}")
    print(f"Contact sheet: {report['all_attempts_contact_sheet']}")
    print(f"Rejected report: {report['rejected_attempt_report']}")
    print(f"Audit report: {Path(report['output_directory']) / 'manual_event_audit_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
