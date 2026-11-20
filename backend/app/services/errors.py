"""Shared errors raised while processing uploaded assessment videos."""


class AssessmentServiceError(Exception):
    """Base error for assessment failures that map to an HTTP response."""


class InvalidVideoError(AssessmentServiceError):
    """The submitted file is not a readable video."""


class MissingPoseModelError(AssessmentServiceError):
    """The configured MediaPipe task model is unavailable."""


class AssessmentProcessingError(AssessmentServiceError):
    """Pose inference or assessment analysis failed."""
