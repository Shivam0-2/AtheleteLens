"""Experimental phase-based push-up repetition analysis.

This module is separate from the production analyzer. It compares an explicit
movement-phase state machine with the production threshold counter and writes
experimental per-frame, event, comparison, and run-metadata files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:  # Package invocation: python -m ai_engine.phase_based_analysis
    from .athletelens_ai.assessments.push_up import (
        DetectorConfig,
        PhaseBasedPushUpDetector,
        PushUpRepetitionCounter,
    )
    from .athletelens_ai.signals import optional_angle, rolling_median
except ImportError:  # Standalone invocation from the ai_engine directory
    from athletelens_ai.assessments.push_up import (
        DetectorConfig,
        PhaseBasedPushUpDetector,
        PushUpRepetitionCounter,
    )
    from athletelens_ai.signals import optional_angle, rolling_median


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "experimental_outputs" / "phase_based"
SIGNALS = ("raw", "rolling_median")


@dataclass
class MeasurementSeries:
    """Frame-aligned measurements loaded from one analyzer CSV."""

    label: str
    source_path: Path
    frame_numbers: list[int]
    timestamps_ms: list[int]
    angles: list[float | None]


def load_measurements(path: Path, label: str) -> MeasurementSeries:
    """Read the frame, timestamp, and selected elbow angle columns."""
    required = {"frame_number", "timestamp_ms", "elbow_angle_deg"}
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        frames: list[int] = []
        timestamps: list[int] = []
        angles: list[float | None] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                frames.append(int(row["frame_number"]))
                timestamps.append(int(row["timestamp_ms"]))
                angles.append(optional_angle(row.get("elbow_angle_deg")))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid data in {path}, row {row_number}: {exc}") from exc
    if not frames:
        raise ValueError(f"No frame measurements found in {path}")
    return MeasurementSeries(label, path.resolve(), frames, timestamps, angles)


def count_production_repetitions(
    values: list[float | None], lower_angle: float, upper_angle: float
) -> int:
    """Run the existing production counter unchanged over a supplied signal."""
    counter = PushUpRepetitionCounter(lower_angle, upper_angle)
    for value in values:
        counter.update(value)
    return counter.repetitions


def run_signal(
    series: MeasurementSeries,
    signal_name: str,
    values: list[float | None],
    config: DetectorConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Generate frame-state and repetition-event rows for one video signal."""
    detector = PhaseBasedPushUpDetector(config)
    frame_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for frame, timestamp, raw_angle, signal_angle in zip(
        series.frame_numbers, series.timestamps_ms, series.angles, values
    ):
        result = detector.update(frame, timestamp, signal_angle)
        frame_rows.append(
            {
                "dataset": series.label,
                "signal": signal_name,
                "frame_number": frame,
                "timestamp_ms": timestamp,
                "raw_elbow_angle_deg": raw_angle if raw_angle is not None else "",
                "analysis_angle_deg": signal_angle if signal_angle is not None else "",
                "movement_state": result["movement_state"],
                "repetition_count": result["repetition_count"],
            }
        )
        for event in result["events"]:
            event_rows.append({"dataset": series.label, "signal": signal_name, **event})

    end_timestamp = series.timestamps_ms[-1]
    frame_interval = (
        series.timestamps_ms[-1] - series.timestamps_ms[-2]
        if len(series.timestamps_ms) > 1
        else 0
    )
    detector.finish(end_timestamp + frame_interval)
    # finish() can emit the final incomplete-cycle rejection after the loop.
    already_written = len(event_rows)
    for event in detector.events[already_written:]:
        event_rows.append({"dataset": series.label, "signal": signal_name, **event})
    return frame_rows, event_rows, detector.repetitions


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_series(
    series: MeasurementSeries,
    config: DetectorConfig,
    smoothing_window: int,
    production_lower: float,
    production_upper: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare both algorithms on raw and rolling-median signals."""
    smoothed = rolling_median(series.angles, smoothing_window)
    frame_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for signal_name, values in (("raw", series.angles), ("rolling_median", smoothed)):
        frames, events, phase_count = run_signal(series, signal_name, values, config)
        frame_rows.extend(frames)
        event_rows.extend(events)
        production_count = count_production_repetitions(
            values, production_lower, production_upper
        )
        comparison_rows.append(
            {
                "dataset": series.label,
                "signal": signal_name,
                "production_lower_angle_deg": production_lower,
                "production_upper_angle_deg": production_upper,
                "production_repetitions": production_count,
                "phase_bottom_angle_deg": config.bottom_angle_deg,
                "phase_top_angle_deg": config.top_angle_deg,
                "phase_hysteresis_deg": config.hysteresis_deg,
                "phase_confirmation_frames": config.confirmation_frames,
                "phase_min_duration_s": config.min_repetition_duration_s,
                "phase_max_duration_s": config.max_repetition_duration_s,
                "phase_max_gap_s": config.max_measurement_gap_s,
                "phase_repetitions": phase_count,
                "phase_minus_production": phase_count - production_count,
            }
        )
    return frame_rows, event_rows, comparison_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimental phase-based push-up analysis and production comparison."
    )
    parser.add_argument(
        "--original-csv", type=Path, default=SCRIPT_DIR / "videos" / "pushup_measurements.csv"
    )
    parser.add_argument(
        "--challenge-csv", type=Path, default=SCRIPT_DIR / "videos" / "pushup1_measurements.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bottom-angle", type=float, default=90.0)
    parser.add_argument("--top-angle", type=float, default=160.0)
    parser.add_argument("--hysteresis", type=float, default=5.0)
    parser.add_argument("--confirmation-frames", type=int, default=2)
    parser.add_argument("--min-repetition-duration", type=float, default=0.25)
    parser.add_argument("--max-repetition-duration", type=float, default=3.0)
    parser.add_argument("--max-measurement-gap", type=float, default=0.25)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--production-lower-angle", type=float, default=90.0)
    parser.add_argument("--production-upper-angle", type=float, default=160.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = DetectorConfig(
        bottom_angle_deg=args.bottom_angle,
        top_angle_deg=args.top_angle,
        hysteresis_deg=args.hysteresis,
        confirmation_frames=args.confirmation_frames,
        min_repetition_duration_s=args.min_repetition_duration,
        max_repetition_duration_s=args.max_repetition_duration,
        max_measurement_gap_s=args.max_measurement_gap,
    )
    try:
        config.validate()
        if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
            raise ValueError("--smoothing-window must be a positive odd number")
        if not (0 <= args.production_lower_angle < args.production_upper_angle <= 180):
            raise ValueError("Production thresholds must satisfy 0 <= lower < upper <= 180")

        series = [
            load_measurements(args.original_csv.expanduser().resolve(), "original"),
            load_measurements(
                args.challenge_csv.expanduser().resolve(), "one_minute_challenge"
            ),
        ]
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        all_frames: list[dict[str, Any]] = []
        all_events: list[dict[str, Any]] = []
        all_comparisons: list[dict[str, Any]] = []
        for item in series:
            frames, events, comparisons = analyze_series(
                item,
                config,
                args.smoothing_window,
                args.production_lower_angle,
                args.production_upper_angle,
            )
            all_frames.extend(frames)
            all_events.extend(events)
            all_comparisons.extend(comparisons)

        write_csv(
            output_dir / "frame_movement_states.csv",
            all_frames,
            [
                "dataset",
                "signal",
                "frame_number",
                "timestamp_ms",
                "raw_elbow_angle_deg",
                "analysis_angle_deg",
                "movement_state",
                "repetition_count",
            ],
        )
        write_csv(
            output_dir / "repetition_events.csv",
            all_events,
            [
                "dataset",
                "signal",
                "event_type",
                "repetition_number",
                "start_timestamp_ms",
                "end_timestamp_ms",
                "duration_seconds",
                "reason",
            ],
        )
        write_csv(
            output_dir / "algorithm_comparison.csv",
            all_comparisons,
            list(all_comparisons[0]),
        )
        metadata = {
            "status": "Experimental; not validated against manually annotated rep intervals.",
            "manual_accuracy_claim": False,
            "config": asdict(config),
            "smoothing_window_frames": args.smoothing_window,
            "production_thresholds": {
                "lower_angle_deg": args.production_lower_angle,
                "upper_angle_deg": args.production_upper_angle,
            },
            "inputs": [
                {
                    "dataset": item.label,
                    "path": str(item.source_path),
                    "sha256": sha256_file(item.source_path),
                    "frames": len(item.frame_numbers),
                    "missing_angle_frames": sum(value is None for value in item.angles),
                }
                for item in series
            ],
            "outputs": {
                "frame_states": str(output_dir / "frame_movement_states.csv"),
                "events": str(output_dir / "repetition_events.csv"),
                "comparison": str(output_dir / "algorithm_comparison.csv"),
            },
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    for row in all_comparisons:
        print(
            f"{row['dataset']} / {row['signal']}: production="
            f"{row['production_repetitions']}, phase-based={row['phase_repetitions']}"
        )
    print(f"Experimental outputs: {output_dir}")
    print("Counts are preliminary and are not accuracy claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
