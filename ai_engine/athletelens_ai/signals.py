"""Shared missing-angle normalization and signal smoothing helpers."""

from __future__ import annotations

import math
import statistics
from typing import Any


def optional_angle(value: Any) -> float | None:
    """Parse an optional angle; preserve blanks and non-finite values as missing."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    angle = float(value)
    return angle if math.isfinite(angle) else None


def rolling_median(
    values: list[float | None], window: int = 7
) -> list[float | None]:
    """Centered rolling median over contiguous valid runs without filling gaps."""
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
            sample = [float(value) for value in values[left:right] if value is not None]
            smoothed[index] = statistics.median(sample)
    return smoothed
