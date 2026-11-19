"""Small shared types for deterministic assessment implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ..pipeline import PoseObservation
from ..video import VideoMetadata


@dataclass(frozen=True)
class AssessmentMeasurement:
    """Measurements and quality values produced for one frame."""

    frame_number: int
    timestamp_ms: int
    values: Mapping[str, float | int | str | None]
    quality: Mapping[str, float | int | str | bool | None] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class AssessmentResult:
    """Structured, assessment-neutral temporal analysis output."""

    assessment_type: str
    summary: Mapping[str, Any]
    events: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    configuration: Mapping[str, Any] = field(default_factory=dict)


class AssessmentProtocol(Protocol):
    """Contract for per-frame measurement followed by temporal analysis."""

    def measure_frame(self, observation: PoseObservation) -> AssessmentMeasurement:
        """Convert one pose observation into assessment-specific measurements."""
        ...

    def analyze(
        self,
        measurements: Sequence[AssessmentMeasurement],
        video: VideoMetadata,
    ) -> AssessmentResult:
        """Apply assessment-specific temporal logic and summarize its result."""
        ...

