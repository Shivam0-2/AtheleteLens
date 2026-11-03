"""Experimental analysis of exported push-up measurement CSV files.

This script compares a threshold state machine on raw and rolling-median
elbow-angle signals. Results are exploratory and are not validated labels.
It reads source CSVs without changing them and writes artifacts to a separate
output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .athletelens_ai.signals import optional_angle, rolling_median
except ImportError:
    from athletelens_ai.signals import optional_angle, rolling_median


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "experimental_outputs"


@dataclass
class MeasurementSeries:
    """Frame-aligned angle and pose-detection data loaded from one CSV."""

    label: str
    source: Path
    frame_numbers: list[int]
    timestamps_ms: list[int]
    angles: list[float | None]
    pose_detected: list[bool]


def parse_thresholds(value: str) -> list[float]:
    """Parse a comma-separated list of angle thresholds."""
    try:
        thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc
    if not thresholds or any(not math.isfinite(item) or not 0 <= item <= 180 for item in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be finite values from 0 to 180")
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare experimental repetition thresholds and angle signals."
    )
    parser.add_argument(
        "--original-csv",
        type=Path,
        default=SCRIPT_DIR / "videos" / "pushup_measurements.csv",
    )
    parser.add_argument(
        "--challenge-csv",
        type=Path,
        default=SCRIPT_DIR / "videos" / "pushup1_measurements.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window", type=int, default=7, help="Odd rolling median window")
    parser.add_argument(
        "--bottom-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("70,80,90,100"),
        help="Comma-separated lower-angle thresholds, degrees",
    )
    parser.add_argument(
        "--top-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("140,150,160,170"),
        help="Comma-separated upper-angle thresholds, degrees",
    )
    parser.add_argument(
        "--extreme-low",
        type=float,
        default=20.0,
        help="Flag angles below this value as suspicious (default: 20)",
    )
    parser.add_argument(
        "--extreme-high",
        type=float,
        default=175.0,
        help="Flag angles above this value as suspicious (default: 175)",
    )
    parser.add_argument(
        "--jump-degrees",
        type=float,
        default=45.0,
        help="Flag consecutive-frame jumps above this size (default: 45)",
    )
    args = parser.parse_args()
    if args.window < 1 or args.window % 2 == 0:
        parser.error("--window must be a positive odd integer")
    if not (0 <= args.extreme_low < args.extreme_high <= 180):
        parser.error("extreme limits must satisfy 0 <= low < high <= 180")
    if args.jump_degrees <= 0 or not math.isfinite(args.jump_degrees):
        parser.error("--jump-degrees must be a positive finite value")
    return args


def load_measurements(path: Path, label: str) -> MeasurementSeries:
    """Load required fields from a production measurement CSV."""
    required = {"frame_number", "timestamp_ms", "pose_detected", "elbow_angle_deg"}
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            missing_columns = required.difference(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"{path} is missing columns: {', '.join(sorted(missing_columns))}"
                )
            frames: list[int] = []
            timestamps: list[int] = []
            angles: list[float | None] = []
            poses: list[bool] = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    frames.append(int(row["frame_number"]))
                    timestamps.append(int(row["timestamp_ms"]))
                    is_pose = row["pose_detected"].strip().lower() in {
                        "true", "1", "yes"
                    }
                    poses.append(is_pose)
                    angles.append(optional_angle(row.get("elbow_angle_deg")))
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid data in {path}, row {row_number}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read input CSV '{path}': {exc}") from exc

    if not frames:
        raise ValueError(f"Input CSV has no frame rows: {path}")
    return MeasurementSeries(label, path.resolve(), frames, timestamps, angles, poses)


def count_repetitions(
    values: list[float | None], bottom: float, top: float
) -> int:
    """Count bottom-then-top cycles, disarming after one count until another bottom."""
    phase = "seeking_bottom"
    repetitions = 0
    for angle in values:
        if angle is None:
            continue
        if phase == "seeking_bottom" and angle <= bottom:
            phase = "bottom_reached"
        elif phase == "bottom_reached" and angle >= top:
            repetitions += 1
            phase = "top_reached"
        elif phase == "top_reached" and angle <= bottom:
            phase = "bottom_reached"
    return repetitions


def inspect_suspicious_angles(
    series: MeasurementSeries,
    extreme_low: float,
    extreme_high: float,
    jump_degrees: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Flag physically invalid, extreme, and abrupt consecutive angle values."""
    suspicious: list[dict[str, Any]] = []
    counts = {
        "outside_0_to_180": 0,
        "extreme_low": 0,
        "extreme_high": 0,
        "sudden_jump": 0,
    }
    previous_angle: float | None = None
    previous_frame: int | None = None
    for index, angle in enumerate(series.angles):
        if angle is None:
            previous_angle = None
            previous_frame = None
            continue
        reasons: list[str] = []
        if angle < 0 or angle > 180:
            reasons.append("outside_0_to_180")
            counts["outside_0_to_180"] += 1
        if angle < extreme_low:
            reasons.append("extreme_low")
            counts["extreme_low"] += 1
        if angle > extreme_high:
            reasons.append("extreme_high")
            counts["extreme_high"] += 1
        if (
            previous_angle is not None
            and previous_frame is not None
            and series.frame_numbers[index] == previous_frame + 1
            and abs(angle - previous_angle) > jump_degrees
        ):
            reasons.append("sudden_jump")
            counts["sudden_jump"] += 1
        if reasons:
            suspicious.append(
                {
                    "dataset": series.label,
                    "frame_number": series.frame_numbers[index],
                    "timestamp_ms": series.timestamps_ms[index],
                    "elbow_angle_deg": angle,
                    "pose_detected": series.pose_detected[index],
                    "reasons": ";".join(reasons),
                }
            )
        previous_angle = angle
        previous_frame = series.frame_numbers[index]
    return suspicious, counts


def percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 3) if denominator else 0.0


def create_angle_plot(
    series: MeasurementSeries,
    smoothed: list[float | None],
    window: int,
    destination: Path,
) -> None:
    """Save an angle-versus-time plot showing raw and smoothed measurements."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [timestamp / 1000.0 for timestamp in series.timestamps_ms]
    raw_values = [math.nan if value is None else value for value in series.angles]
    smooth_values = [math.nan if value is None else value for value in smoothed]

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(times, raw_values, color="#7c8da6", alpha=0.48, linewidth=0.8, label="Raw")
    axis.plot(
        times,
        smooth_values,
        color="#1769aa",
        linewidth=1.5,
        label=f"Rolling median (window={window})",
    )
    axis.set(
        title=f"{series.label}: elbow angle versus time (preliminary)",
        xlabel="Time (seconds)",
        ylabel="Elbow angle (degrees)",
        ylim=(0, 185),
    )
    axis.grid(True, alpha=0.22)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def run_experiment(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Analyze both series and export plots, threshold counts, and quality data."""
    original = load_measurements(args.original_csv.expanduser().resolve(), "original")
    challenge = load_measurements(args.challenge_csv.expanduser().resolve(), "one_minute_challenge")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict[str, Any]] = []
    suspicious_rows: list[dict[str, Any]] = []
    quality_report: dict[str, Any] = {
        "status": "Preliminary exploratory analysis; only two video CSVs are included.",
        "thresholds_selected": False,
        "rolling_median_window_frames": args.window,
        "suspicious_angle_rules": {
            "physical_range_degrees": [0, 180],
            "extreme_low_below_degrees": args.extreme_low,
            "extreme_high_above_degrees": args.extreme_high,
            "consecutive_frame_jump_above_degrees": args.jump_degrees,
        },
        "videos": [],
    }

    for series in (original, challenge):
        smoothed = rolling_median(series.angles, args.window)
        plot_path = output_dir / f"{series.label}_angle_vs_time.png"
        create_angle_plot(series, smoothed, args.window, plot_path)

        missing_angles = sum(angle is None for angle in series.angles)
        undetected_poses = sum(not detected for detected in series.pose_detected)
        suspicious, suspicious_counts = inspect_suspicious_angles(
            series, args.extreme_low, args.extreme_high, args.jump_degrees
        )
        suspicious_rows.extend(suspicious)
        quality_report["videos"].append(
            {
                "dataset": series.label,
                "source_csv": str(series.source),
                "total_frames": len(series.angles),
                "missing_angle_frames": missing_angles,
                "missing_angle_percentage": percentage(missing_angles, len(series.angles)),
                "pose_detected_frames": sum(series.pose_detected),
                "pose_detection_percentage": percentage(
                    len(series.angles) - undetected_poses, len(series.angles)
                ),
                "suspicious_measurement_rows": len(suspicious),
                "suspicious_measurement_counts_by_rule": suspicious_counts,
                "plot": str(plot_path),
            }
        )

        signals = (("raw", series.angles), ("rolling_median", smoothed))
        for bottom in args.bottom_thresholds:
            for top in args.top_thresholds:
                if bottom >= top:
                    continue
                for signal_name, values in signals:
                    comparison_rows.append(
                        {
                            "dataset": series.label,
                            "signal": signal_name,
                            "smoothing_window_frames": args.window if signal_name == "rolling_median" else "",
                            "bottom_threshold_deg": bottom,
                            "top_threshold_deg": top,
                            "detected_repetitions": count_repetitions(values, bottom, top),
                        }
                    )

    if not comparison_rows:
        raise ValueError("No valid threshold pairs: every bottom threshold must be below a top threshold")

    comparison_path = output_dir / "threshold_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    suspicious_path = output_dir / "suspicious_measurements.csv"
    with suspicious_path.open("w", newline="", encoding="utf-8") as destination:
        fields = ["dataset", "frame_number", "timestamp_ms", "elbow_angle_deg", "pose_detected", "reasons"]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(suspicious_rows)

    quality_report["outputs"] = {
        "threshold_comparison_csv": str(comparison_path),
        "suspicious_measurements_csv": str(suspicious_path),
    }
    quality_path = output_dir / "quality_and_suspicion.json"
    quality_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
    return comparison_rows, quality_report["videos"]


def main() -> int:
    args = parse_args()
    try:
        comparison_rows, videos = run_experiment(args)
    except (OSError, ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    for video in videos:
        print(
            f"{video['dataset']}: {video['total_frames']} frames; "
            f"missing angles {video['missing_angle_percentage']}%; "
            f"pose detected {video['pose_detection_percentage']}%; "
            f"suspicious rows {video['suspicious_measurement_rows']}"
        )
        print(f"  Plot: {video['plot']}")
    print(f"Threshold comparison rows: {len(comparison_rows)}")
    print(
        "All comparisons are preliminary; no thresholds are selected and no "
        "counts are validated against manual annotations."
    )
    print(f"Output directory: {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
