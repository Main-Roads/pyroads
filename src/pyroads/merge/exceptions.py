"""Custom exceptions used by the pyroads.merge package."""

from __future__ import annotations


class MergeSegmentsError(RuntimeError):
    """Base class for errors raised by pyroads.merge."""


class InvalidJoinConfigurationError(MergeSegmentsError):
    """Raised when join keys or interval columns are missing or malformed."""


class InvalidDataFrameError(TypeError, MergeSegmentsError):
    """Raised when inputs are not well-formed pandas DataFrames."""


class InvalidSegmentError(MergeSegmentsError):
    """Raised when segment geometry or metadata is invalid."""


class ZeroLengthSegmentError(InvalidSegmentError):
    """Raised when either data or target contains zero-length intervals."""


class DuplicateLabelError(MergeSegmentsError):
    """Raised when duplicated indices or column labels are detected."""


class OutputCollisionError(MergeSegmentsError):
    """Raised when a requested output column already exists in the target."""


class InvalidAggregationError(TypeError, MergeSegmentsError):
    """Raised when an aggregation is incompatible with the source column dtype."""


class PercentileConfigurationError(MergeSegmentsError):
    """Raised when length-weighted percentile arguments are invalid."""
