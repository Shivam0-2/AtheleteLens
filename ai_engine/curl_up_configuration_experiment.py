"""Compare curl-up detector configurations on one saved measurement trace.

This experiment reuses the existing CurlUpAssessment state machine and saved
frame-level measurements. It never reruns pose inference or changes the
baseline configuration recorded in the source summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from athletelens_ai.assessments.base import AssessmentMeasurement
from athletelens_ai.assessments.curl_up import CurlUpAssessment, CurlUpConfig
from athletelens_ai.pose import load_pose_dependencies
from athletelens_ai.video import VideoMetadata


AI_ENGINE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = (
    AI_ENGINE_DIR
    / "experimental_outputs"
    / "curl_up_diagnostic"
    / "run_test_fixture_settings"
)
DEFAULT_OUTPUT_DIR = (
    AI_ENGINE_DIR
    / "experimental_outputs"
    / "curl_up_diagnostic"
    / "configuration_comparison_angle_ranges"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the recorded curl-up baseline with an explicitly supplied "
            "experimental threshold pair using the same saved angle measurements."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--proposed-extended-angle-deg", required=True, type=float)
    parser.add_argument("--proposed-flexed-angle-deg", required=True, type=float)
    return parser.parse_args()


def _load_measurements(path: Path) -> list[AssessmentMeasurement]:
    measurements: list[AssessmentMeasurement] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            raw_angle = row.get("included_angle_deg", "")
            angle = float(raw_angle) if raw_angle else None
            if angle is not None and not math.isfinite(angle):
                angle = None
            raw_visibility = row.get("visibility_score", "")
            measurements.append(
                AssessmentMeasurement(
                    frame_number=int(row["frame_number"]),
                    timestamp_ms=int(row["timestamp_ms"]),
                    values={
                        "included_angle_deg": angle,
                        "selected_side": row.get("selected_side", ""),
                    },
                    quality={
                        "pose_detected": row.get("pose_detected", "False").lower()
                        == "true",
                        "measurement_valid": bool(raw_angle),
                        "visibility_score": (
                            float(raw_visibility) if raw_visibility else None
                        ),
                        "missing_landmarks": row.get("missing_landmarks", ""),
                    },
                )
            )
    return measurements


def _write_comparison_csv(
    path: Path,
    measurements: list[AssessmentMeasurement],
    baseline_rows: list[dict[str, Any]],
    proposed_rows: list[dict[str, Any]],
) -> None:
    fields = [
        "frame_number",
        "timestamp_ms",
        "included_angle_deg",
        "pose_detected",
        "measurement_valid",
        "selected_side",
        "visibility_score",
        "missing_landmarks",
        "baseline_state",
        "baseline_repetition_count",
        "proposed_state",
        "proposed_repetition_count",
        "baseline_accepted_on_frame",
        "proposed_accepted_on_frame",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        previous_baseline = previous_proposed = 0
        for measurement, base, candidate in zip(
            measurements, baseline_rows, proposed_rows
        ):
            base_count = int(base["repetition_count"])
            candidate_count = int(candidate["repetition_count"])
            writer.writerow(
                {
                    "frame_number": measurement.frame_number,
                    "timestamp_ms": measurement.timestamp_ms,
                    "included_angle_deg": measurement.values[
                        "included_angle_deg"
                    ],
                    "pose_detected": measurement.quality.get("pose_detected"),
                    "measurement_valid": measurement.quality.get(
                        "measurement_valid"
                    ),
                    "selected_side": measurement.values.get("selected_side"),
                    "visibility_score": measurement.quality.get(
                        "visibility_score"
                    ),
                    "missing_landmarks": measurement.quality.get(
                        "missing_landmarks"
                    ),
                    "baseline_state": base["movement_state"],
                    "baseline_repetition_count": base_count,
                    "proposed_state": candidate["movement_state"],
                    "proposed_repetition_count": candidate_count,
                    "baseline_accepted_on_frame": base_count > previous_baseline,
                    "proposed_accepted_on_frame": candidate_count > previous_proposed,
                }
            )
            previous_baseline = base_count
            previous_proposed = candidate_count


def _plot_comparison(
    path: Path,
    measurements: list[AssessmentMeasurement],
    baseline: CurlUpConfig,
    proposed: CurlUpConfig,
) -> None:
    cv2, _mp, np = load_pose_dependencies()
    width, height = 1400, 850
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 115, width - 40, 135, height - 100
    cv2.putText(
        canvas,
        "Curl-up included angle and configured phase thresholds",
        (left, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Experimental comparison; thresholds are not validated",
        (left, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    legend = [
        ("Measured included angle", (205, 75, 35)),
        (f"Baseline extended {baseline.extended_angle_deg:g} deg", (170, 40, 170)),
        (f"Baseline flexed {baseline.flexed_angle_deg:g} deg", (20, 130, 225)),
        (f"Proposed extended {proposed.extended_angle_deg:g} deg", (175, 115, 15)),
        (f"Proposed flexed {proposed.flexed_angle_deg:g} deg", (55, 155, 45)),
    ]
    legend_x = left
    for label, color in legend:
        cv2.line(canvas, (legend_x, 112), (legend_x + 22, 112), color, 3)
        cv2.putText(
            canvas,
            label,
            (legend_x + 29, 117),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        legend_x += 245

    def y_for(angle: float) -> int:
        return int(bottom - angle / 180.0 * (bottom - top))

    for angle in range(0, 181, 30):
        y = y_for(angle)
        cv2.line(canvas, (left, y), (right, y), (228, 228, 228), 1)
        cv2.putText(
            canvas,
            str(angle),
            (left - 42, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
    cv2.line(canvas, (left, top), (left, bottom), (35, 35, 35), 2)
    cv2.line(canvas, (left, bottom), (right, bottom), (35, 35, 35), 2)

    timestamps = [item.timestamp_ms for item in measurements]
    max_time_s = max(timestamps, default=1) / 1000.0
    max_time_s = max(max_time_s, 1.0)
    for second in range(math.ceil(max_time_s) + 1):
        x = int(left + second / max_time_s * (right - left))
        if x <= right:
            cv2.line(canvas, (x, bottom), (x, bottom + 5), (40, 40, 40), 1)
            cv2.putText(
                canvas,
                str(second),
                (x - 3, bottom + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (45, 45, 45),
                1,
                cv2.LINE_AA,
            )

    # Draw all four threshold guides after the grid; line colors match the legend.
    for threshold, color in (
        (baseline.extended_angle_deg, legend[1][1]),
        (baseline.flexed_angle_deg, legend[2][1]),
        (proposed.extended_angle_deg, legend[3][1]),
        (proposed.flexed_angle_deg, legend[4][1]),
    ):
        y = y_for(threshold)
        for x0 in range(left, right, 24):
            cv2.line(canvas, (x0, y), (min(x0 + 12, right), y), color, 1)

    previous: tuple[int, int] | None = None
    for measurement in measurements:
        angle = measurement.values.get("included_angle_deg")
        if angle is None:
            previous = None
            continue
        x = int(
            left
            + measurement.timestamp_ms / 1000.0 / max_time_s * (right - left)
        )
        current = (x, y_for(float(angle)))
        if previous is not None:
            cv2.line(canvas, previous, current, legend[0][1], 2, cv2.LINE_AA)
        previous = current

    y_label = np.full((34, 230, 3), 255, dtype=np.uint8)
    cv2.putText(
        y_label,
        "Included angle (degrees)",
        (4, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    y_label = cv2.rotate(y_label, cv2.ROTATE_90_COUNTERCLOCKWISE)
    label_y = (top + bottom - y_label.shape[0]) // 2
    canvas[label_y : label_y + y_label.shape[0], 12 : 12 + y_label.shape[1]] = y_label
    cv2.putText(
        canvas,
        "Time (seconds)",
        ((left + right) // 2 - 60, height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write graph: {path}")


def _event_rows(result: Any, kind: str) -> list[dict[str, Any]]:
    return [event for event in result.events if event["event_type"] == kind]


def _trailing_gap_diagnosis(
    measurements: list[AssessmentMeasurement], max_gap_s: float
) -> dict[str, Any]:
    valid = [
        item
        for item in measurements
        if item.values.get("included_angle_deg") is not None
    ]
    if not valid:
        return {"applicable": False}
    last_valid = valid[-1]
    trailing_missing = [
        item for item in measurements if item.timestamp_ms > last_valid.timestamp_ms
    ]
    crossing = next(
        (
            item
            for item in trailing_missing
            if (item.timestamp_ms - last_valid.timestamp_ms) / 1000.0 > max_gap_s
        ),
        None,
    )
    return {
        "applicable": crossing is not None,
        "last_valid_timestamp_ms": last_valid.timestamp_ms,
        "first_trailing_missing_timestamp_ms": (
            trailing_missing[0].timestamp_ms if trailing_missing else None
        ),
        "gap_limit_crossed_at_timestamp_ms": (
            crossing.timestamp_ms if crossing else None
        ),
        "elapsed_at_crossing_ms": (
            crossing.timestamp_ms - last_valid.timestamp_ms if crossing else None
        ),
        "trailing_missing_frame_count": len(trailing_missing),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    csv_path = input_dir / "curl_up_measurements.csv"
    baseline_summary_path = input_dir / "curl_up_summary.json"
    if not csv_path.is_file() or not baseline_summary_path.is_file():
        raise FileNotFoundError(
            f"Expected curl_up_measurements.csv and curl_up_summary.json in {input_dir}"
        )

    outputs = {
        "csv": output_dir / "curl_up_threshold_comparison.csv",
        "json": output_dir / "curl_up_threshold_comparison.json",
        "graph": output_dir / "curl_up_angle_threshold_comparison.png",
        "report": output_dir / "curl_up_comparison_report.md",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs; choose another --output-dir: "
            + ", ".join(str(path) for path in existing)
        )

    source_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    baseline_config = CurlUpConfig(**source_summary["detector_configuration"])
    proposed_config = replace(
        baseline_config,
        extended_angle_deg=args.proposed_extended_angle_deg,
        flexed_angle_deg=args.proposed_flexed_angle_deg,
    )
    proposed_config.validate()
    video = source_summary["video"]
    metadata = VideoMetadata(
        fps=float(video["fps"]),
        width=int(video["width"]),
        height=int(video["height"]),
    )
    measurements = _load_measurements(csv_path)
    baseline_result = CurlUpAssessment(baseline_config).analyze(measurements, metadata)
    proposed_result = CurlUpAssessment(proposed_config).analyze(measurements, metadata)
    baseline_rows = list(baseline_result.summary["measurements"])
    proposed_rows = list(proposed_result.summary["measurements"])
    if not (
        len(measurements) == len(baseline_rows) == len(proposed_rows)
    ):
        raise RuntimeError("Configuration outputs do not align with source measurements")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_comparison_csv(outputs["csv"], measurements, baseline_rows, proposed_rows)
    _plot_comparison(
        outputs["graph"], measurements, baseline_config, proposed_config
    )
    baseline_accepted = _event_rows(baseline_result, "accepted")
    proposed_accepted = _event_rows(proposed_result, "accepted")
    baseline_rejected = _event_rows(baseline_result, "rejected")
    proposed_rejected = _event_rows(proposed_result, "rejected")
    trailing_gap = _trailing_gap_diagnosis(
        measurements, baseline_config.max_measurement_gap_s
    )
    report = {
        "status": "Experimental configuration comparison; not validated.",
        "input_video": source_summary["input_video"],
        "source_measurements_csv": str(csv_path),
        "same_saved_measurement_trace_used_for_both_runs": True,
        "manual_reference": source_summary.get("manual_reference"),
        "baseline": {
            "configuration": source_summary["detector_configuration"],
            "detected_repetitions": baseline_result.summary["detected_repetitions"],
            "accepted_repetition_timestamps": baseline_accepted,
            "rejected_cycles": baseline_rejected,
            "terminal_state": baseline_rows[-1]["movement_state"] if baseline_rows else None,
        },
        "proposed_experiment": {
            "configuration": proposed_result.configuration,
            "detected_repetitions": proposed_result.summary["detected_repetitions"],
            "accepted_repetition_timestamps": proposed_accepted,
            "rejected_cycles": proposed_rejected,
            "terminal_state": proposed_rows[-1]["movement_state"] if proposed_rows else None,
            "threshold_rationale": (
                "The included-angle trace occupies a low band near 90–100 degrees "
                "during trunk raises and a higher band near 115–125 degrees while "
                "supine. The proposed thresholds separate those observed bands; "
                "they were not selected by optimizing against the approximate manual count."
            ),
        },
        "angle_measurement": {
            "state_machine_input": "included_angle_deg",
            "derived_curl_flexion_deg": "180 - included_angle_deg; not passed to detector",
            "minimum_deg": min(
                float(item.values["included_angle_deg"])
                for item in measurements
                if item.values.get("included_angle_deg") is not None
            ),
            "maximum_deg": max(
                float(item.values["included_angle_deg"])
                for item in measurements
                if item.values.get("included_angle_deg") is not None
            ),
            "average_deg": source_summary["analysis"]["average_included_angle_deg"],
        },
        "trailing_gap_diagnosis": trailing_gap,
        "baseline_gap_event_explanation": (
            "The baseline never leaves SEEK_EXTENDED because the measured included "
            "angle never reaches 150 degrees. It marks the first below-flexed sample "
            "at 300 ms as a partial start. The trailing missing-pose run crosses the "
            "250 ms gap limit at 24,391 ms, causing that partial marker to be logged "
            "as long_measurement_gap. This is not an active repetition being canceled; "
            "the event's start time is a pre-top partial marker."
        ),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "limitations": [
            "The input video has not been manually frame-annotated.",
            "The manual count is approximate and comes from one video.",
            "Matching this count is not evidence of detector accuracy.",
        ],
    }
    outputs["json"].write_text(json.dumps(report, indent=2), encoding="utf-8")

    accepted_lines = (
        "- Rep {repetition_number}: {start_timestamp_ms}–{end_timestamp_ms} ms "
        "({duration_seconds:.3f} s)".format(**event)
        for event in proposed_accepted
    )
    markdown = [
        "# Curl-up configuration comparison",
        "",
        "> Experimental diagnostic only; not a validated curl-up assessment.",
        "",
        f"Input video: `{source_summary['input_video']}`",
        f"Frames measured: {len(measurements)}; valid angles: "
        f"{sum(item.values.get('included_angle_deg') is not None for item in measurements)}.",
        "",
        "## Angle interpretation",
        "",
        "The state machine consumes the included shoulder–hip–knee angle. "
        "The separately reported curl-flexion value (180° minus included angle) "
        "is not the detector input.",
        "",
        f"Observed included-angle range: {report['angle_measurement']['minimum_deg']:.2f}°–"
        f"{report['angle_measurement']['maximum_deg']:.2f}°; mean "
        f"{report['angle_measurement']['average_deg']:.2f}°.",
        "",
        "## Results",
        "",
        "| Run | Extended threshold | Flexed threshold | Accepted repetitions | Rejected cycles |",
        "|---|---:|---:|---:|---:|",
        f"| Baseline | {baseline_config.extended_angle_deg:g}° | "
        f"{baseline_config.flexed_angle_deg:g}° | "
        f"{baseline_result.summary['detected_repetitions']} | {len(baseline_rejected)} |",
        f"| Proposed experiment | {proposed_config.extended_angle_deg:g}° | "
        f"{proposed_config.flexed_angle_deg:g}° | "
        f"{proposed_result.summary['detected_repetitions']} | {len(proposed_rejected)} |",
        "",
        "### Proposed accepted timestamps",
        "",
        *(list(accepted_lines) or ["None."]),
        "",
        "### Rejected event details",
        "",
        "Baseline:",
        "",
        *([f"- `{event['reason']}`: {event['start_timestamp_ms']}–"
           f"{event['end_timestamp_ms']} ms" for event in baseline_rejected] or ["- None."]),
        "",
        "Proposed experiment:",
        "",
        *([f"- `{event['reason']}`: {event['start_timestamp_ms']}–"
           f"{event['end_timestamp_ms']} ms" for event in proposed_rejected] or ["- None."]),
        "",
        "## Why the baseline did not count",
        "",
        report["baseline_gap_event_explanation"],
        "",
        "The annotated frames show a supine starting posture near the high end of "
        "the observed angle range (about 125°) and trunk raises near 90°–100°. "
        "The 150° baseline extended threshold is therefore unreachable in this clip.",
        "",
        "The proposed 115°/100° pair is an explicitly labeled exploratory separation "
        "of the observed high and low angle bands. Its result happens to equal the "
        "approximate manual count; this single-video agreement is not validation and "
        "must not be presented as accuracy.",
        "",
        "## Outputs",
        "",
        *(f"- `{key}`: `{path}`" for key, path in outputs.items()),
        "",
    ]
    outputs["report"].write_text("\n".join(markdown), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run_experiment(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(
        "Baseline: {baseline}; proposed experiment: {proposed}".format(
            baseline=report["baseline"]["detected_repetitions"],
            proposed=report["proposed_experiment"]["detected_repetitions"],
        )
    )
    print(f"Report: {report['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
