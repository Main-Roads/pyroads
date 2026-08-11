"""Public package interface for pyroads.merge.

Exposes the primary merge helpers and aggregation utilities so callers can
import them directly from :mod:`pyroads.merge` instead of navigating the
internal module layout.
"""

from importlib.metadata import PackageNotFoundError, version

from .merge import (  # noqa: F401 - re-exported for convenience
    Action,
    Aggregation,
    configure_performance_logger,
    is_numba_available,
    on_slk_intervals,
    on_slk_intervals_auto,
    on_slk_intervals_fallback,
    on_slk_intervals_numba,
    on_slk_intervals_optimized,
)
from ._polars_merge import is_polars_available, on_slk_intervals_polars  # noqa: F401
from ._dask_merge import is_dask_available, on_slk_intervals_dask  # noqa: F401

try:
    __version__ = version("pyroads.merge")
except PackageNotFoundError:  # pragma: no cover - during local editing
    __version__ = "1.2.0"

__all__ = [
    "Action",
    "Aggregation",
    "configure_performance_logger",
    "is_numba_available",
    "is_polars_available",
    "is_dask_available",
    "on_slk_intervals",
    "on_slk_intervals_auto",
    "on_slk_intervals_fallback",
    "on_slk_intervals_numba",
    "on_slk_intervals_optimized",
    "on_slk_intervals_polars",
    "on_slk_intervals_dask",
    "__version__",
]
