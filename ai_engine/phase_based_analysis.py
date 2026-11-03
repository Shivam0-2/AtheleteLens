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
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:  # Package invocation: python -m ai_engine.phase_based_analysis
    from .pushup_analysis import PushUpRepetitionCounter
except ImportError:  # Standalone invocation from the ai_engine directory
    from pushup_analysis import PushUpRepetitionCounter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "experimental_outputs" / "phase_based"
SIGNALS = ("raw", "rolling_median")


@dataclass(frozen=True)
class DetectorConfig:
    """Configurable phase thresholds, confirmation, duration, and gap limits."""

    bottom_angle_deg: float = 90.0
    top_angle_deg: float = 160.0
    hysteresis_deg: float = 5.0
    confirmation_frames: int = 2
    min_repetition_duration_s: float = 0.25
    max_repetition_duration_s: float = 3.0
    max_measurement_gap_s: float = 0.25

    def validate(self) -> None:
        if not (0 <= self.bottom_angle_deg < self.top_angle_deg <= 180):
            raise ValueError("Require 0 <= bottom angle < top angle <= 180 degrees")
        if not math.isfinite(self.hysteresis_deg) or self.hysteresis_deg < 0:
            raise ValueError("Hysteresis must be a finite non-negative angle")
        if self.bottom_angle_deg + self.hysteresis_deg >= self.top_angle_deg - self.hysteresis_deg:
            raise ValueError("Hysteresis bands overlap; widen the threshold gap or reduce hysteresis")
        if self.confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least 1")
        if (
            not math.isfinite(self.min_repetition_duration_s)
            or not math.isfinite(self.max_repetition_duration_s)
            or self.min_repetition_duration_s < 0
            or self.max_repetition_duration_s <= self.min_repetition_duration_s
        ):
            raise ValueError("Repetition durations must satisfy 0 <= min < max")
        if not math.isfinite(self.max_measurement_gap_s) or self.max_measurement_gap_s <= 0:
            raise ValueError("max_measurement_gap_s must be positive and finite")


@dataclass
class MeasurementSeries:
    """Frame-aligned measurements loaded from one analyzer CSV."""

    label: str
    source_path: Path
    frame_numbers: list[int]
    timestamps_ms: list[int]
    angles: list[float | None]


def rolling_median(values: list[float | None], window: int = 7) -> list[float | None]:
    """Centered median filter over contiguous valid-angle runs; no gap filling."""
    if window < 1 or window % 2 == 0:
        raise ValueError("Rolling-median window must be a positive odd number")
    radius = window // 2
    smoothed: list[float | None] = [None] * len(values)
    cursor = 0
    while cursor < len(values):
        while cursor < len(values) and values[cursor] is None:
            cursor += 1
        if cursor == len(values):
            break
        start = cursor
        while cursor < len(values) and values[cursor] is not None:
            cursor += 1
        end = cursor
        for index in range(start, end):
            left = max(start, index - radius)
            right = min(end, index + radius + 1)
            smoothed[index] = statistics.median(
                float(value) for value in values[left:right] if value is not None
            )
    return smoothed


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
                value = (row.get("elbow_angle_deg") or "").strip()
                angle = float(value) if value else None
                angles.append(angle if angle is not None and math.isfinite(angle) else None)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid data in {path}, row {row_number}: {exc}") from exc
    if not frames:
        raise ValueError(f"No frame measurements found in {path}")
    return MeasurementSeries(label, path.resolve(), frames, timestamps, angles)


class PhaseBasedPushUpDetector:
    """Count only observed TOP → DESCENDING → BOTTOM → ASCENDING → TOP cycles."""

    SEEK_TOP = "SEEK_TOP"
    TOP = "TOP"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"

    def __init__(self, config: DetectorConfig):
        config.validate()
        self.config = config
        self.state = self.SEEK_TOP
        self.repetitions = 0
        self.events: list[dict[str, Any]] = []
        self._last_valid_timestamp_ms: int | None = None
        self._cycle_start_ms: int | None = None
        self._seek_partial_start_ms: int | None = None
        self._has_seen_top = False
        self._confirmation_name: str | None = None
        self._confirmation_count = 0
        self._confirmation_start_ms: int | None = None

    def _clear_confirmation(self) -> None:
        self._confirmation_name = None
        self._confirmation_count = 0
        self._confirmation_start_ms = None

    def _confirm(self, name: str, condition: bool, timestamp_ms: int) -> int | None:
        """Return the first timestamp once a condition holds for N valid frames."""
        if not condition:
            self._clear_confirmation()
            return None
        if self._confirmation_name == name:
            self._confirmation_count += 1
        else:
            self._confirmation_name = name
            self._confirmation_count = 1
            self._confirmation_start_ms = timestamp_ms
        if self._confirmation_count >= self.config.confirmation_frames:
            first_timestamp = self._confirmation_start_ms
            self._clear_confirmation()
            return first_timestamp
        return None

    def _emit_rejection(self, reason: str, end_timestamp_ms: int) -> None:
        start = self._cycle_start_ms
        if start is None:
            start = self._seek_partial_start_ms
        self.events.append(
            {
                "event_type": "rejected",
                "repetition_number": "",
                "start_timestamp_ms": start if start is not None else "",
                "end_timestamp_ms": end_timestamp_ms,
                "duration_seconds": (
                    round((end_timestamp_ms - start) / 1000.0, 3)
                    if start is not None
                    else ""
                ),
                "reason": reason,
            }
        )

    def _reset_to_seek_top(self) -> None:
        self.state = self.SEEK_TOP
        self._cycle_start_ms = None
        self._seek_partial_start_ms = None
        self._clear_confirmation()

    def _reject_cycle(self, reason: str, end_timestamp_ms: int) -> None:
        self._emit_rejection(reason, end_timestamp_ms)
        self._reset_to_seek_top()

    def _long_gap(self, timestamp_ms: int) -> bool:
        if self._last_valid_timestamp_ms is None:
            return False
        elapsed_s = (timestamp_ms - self._last_valid_timestamp_ms) / 1000.0
        if elapsed_s <= self.config.max_measurement_gap_s:
            return False
        if self.state in {self.DESCENDING, self.BOTTOM, self.ASCENDING}:
            self._reject_cycle("long_measurement_gap", timestamp_ms)
        elif self._seek_partial_start_ms is not None:
            self._emit_rejection("long_measurement_gap", timestamp_ms)
            self._reset_to_seek_top()
        else:
            self._reset_to_seek_top()
        return True

    def _record_completion(self, start_ms: int, end_ms: int) -> None:
        duration_s = (end_ms - start_ms) / 1000.0
        if duration_s < self.config.min_repetition_duration_s:
            self._reject_cycle("too_short", end_ms)
            return
        if duration_s > self.config.max_repetition_duration_s:
            self._reject_cycle("excessively_long", end_ms)
            return
        self.repetitions += 1
        self.events.append(
            {
                "event_type": "completed",
                "repetition_number": self.repetitions,
                "start_timestamp_ms": start_ms,
                "end_timestamp_ms": end_ms,
                "duration_seconds": round(duration_s, 3),
                "reason": "",
            }
        )
        self.state = self.TOP
        self._cycle_start_ms = None
        self._clear_confirmation()

    def update(
        self, frame_number: int, timestamp_ms: int, angle_deg: float | None
    ) -> dict[str, Any]:
        """Consume one frame and return its resulting movement state."""
        event_start = len(self.events)
        if angle_deg is not None and not math.isfinite(angle_deg):
            angle_deg = None

        if angle_deg is None:
            self._clear_confirmation()
            if self._last_valid_timestamp_ms is not None:
                gap_s = (timestamp_ms - self._last_valid_timestamp_ms) / 1000.0
                if gap_s > self.config.max_measurement_gap_s:
                    if self.state in {self.DESCENDING, self.BOTTOM, self.ASCENDING}:
                        self._reject_cycle("long_measurement_gap", timestamp_ms)
                    elif self._seek_partial_start_ms is not None:
                        self._emit_rejection("long_measurement_gap", timestamp_ms)
                        self._reset_to_seek_top()
                    else:
                        self._reset_to_seek_top()
            return self._frame_result(frame_number, timestamp_ms, None, event_start)

        self._long_gap(timestamp_ms)
        self._last_valid_timestamp_ms = timestamp_ms

        if (
            self._cycle_start_ms is not None
            and (timestamp_ms - self._cycle_start_ms) / 1000.0
            > self.config.max_repetition_duration_s
        ):
            self._reject_cycle("excessively_long", timestamp_ms)

        bottom = self.config.bottom_angle_deg
        bottom_exit = bottom + self.config.hysteresis_deg
        top = self.config.top_angle_deg
        top_exit = top - self.config.hysteresis_deg

        if self.state == self.SEEK_TOP:
            confirmed = self._confirm("top", angle_deg >= top, timestamp_ms)
            if angle_deg <= bottom and self._seek_partial_start_ms is None:
                self._seek_partial_start_ms = timestamp_ms
            if confirmed is not None:
                if self._seek_partial_start_ms is not None:
                    self.events.append(
                        {
                            "event_type": "rejected",
                            "repetition_number": "",
                            "start_timestamp_ms": self._seek_partial_start_ms,
                            "end_timestamp_ms": confirmed,
                            "duration_seconds": round(
                                (confirmed - self._seek_partial_start_ms) / 1000.0, 3
                            ),
                            "reason": (
                                "video_started_mid_repetition"
                                if not self._has_seen_top
                                else "no_confirmed_top_before_descent"
                            ),
                        }
                    )
                self.state = self.TOP
                self._seek_partial_start_ms = None
                self._has_seen_top = True

        elif self.state == self.TOP:
            confirmed = self._confirm("leave_top", angle_deg <= top_exit, timestamp_ms)
            if confirmed is not None:
                self.state = self.DESCENDING
                self._cycle_start_ms = confirmed

        elif self.state == self.DESCENDING:
            if angle_deg <= bottom:
                confirmed = self._confirm("reach_bottom", True, timestamp_ms)
                if confirmed is not None:
                    self.state = self.BOTTOM
            elif angle_deg >= top:
                recovered = self._confirm("abort_descent", True, timestamp_ms)
                if recovered is not None:
                    self._reject_cycle("bottom_not_reached", recovered)
            else:
                self._clear_confirmation()

        elif self.state == self.BOTTOM:
            confirmed = self._confirm("leave_bottom", angle_deg >= bottom_exit, timestamp_ms)
            if confirmed is not None:
                self.state = self.ASCENDING

        elif self.state == self.ASCENDING:
            if angle_deg >= top:
                reached_top = self._confirm("reach_top", True, timestamp_ms)
                if reached_top is not None:
                    start_ms = (
                        self._cycle_start_ms
                        if self._cycle_start_ms is not None
                        else reached_top
                    )
                    self._record_completion(start_ms, reached_top)
            elif angle_deg <= bottom:
                returned_to_bottom = self._confirm(
                    "return_to_bottom", True, timestamp_ms
                )
                if returned_to_bottom is not None:
                    self.state = self.BOTTOM
            else:
                self._clear_confirmation()

        return self._frame_result(frame_number, timestamp_ms, angle_deg, event_start)

    def _frame_result(
        self,
        frame_number: int,
        timestamp_ms: int,
        angle_deg: float | None,
        event_start: int,
    ) -> dict[str, Any]:
        return {
            "frame_number": frame_number,
            "timestamp_ms": timestamp_ms,
            "angle_deg": angle_deg if angle_deg is not None else "",
            "movement_state": self.state,
            "repetition_count": self.repetitions,
            "events": self.events[event_start:],
        }

    def finish(self, timestamp_ms: int) -> None:
        """Record an incomplete cycle at end-of-video without counting it."""
        if self.state in {self.DESCENDING, self.BOTTOM, self.ASCENDING}:
            self._reject_cycle("end_of_video_incomplete_cycle", timestamp_ms)
        elif self.state == self.SEEK_TOP and self._seek_partial_start_ms is not None:
            reason = (
                "video_started_mid_repetition"
                if not self._has_seen_top
                else "no_confirmed_top_before_descent"
            )
            self._emit_rejection(reason, timestamp_ms)
            self._reset_to_seek_top()


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
