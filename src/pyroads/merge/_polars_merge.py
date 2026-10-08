"""Polars-backed interval merge implementation.

Validation and dataframe glue are Polars-native. Group ids are computed in
Polars, and the merge itself runs through the same grouped native kernel as
the pandas backend (see :mod:`._grouped`), so both backends return identical
results. The kernel releases the GIL and parallelises across groups with Rayon.

Requires: polars, numba (numba is already a core dependency of the package).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .exceptions import (
    DuplicateLabelError,
    InvalidAggregationError,
    InvalidDataFrameError,
    InvalidJoinConfigurationError,
    OutputCollisionError,
    ZeroLengthSegmentError,
)
from ._numba_merge import AGG_KEEP_LONGEST, NUMBA_AVAILABLE, _get_agg_type_code

if TYPE_CHECKING:
    import polars as pl

    POLARS_AVAILABLE = True
else:
    try:
        import polars as pl

        POLARS_AVAILABLE = True
    except ImportError:  # pragma: no cover - optional dependency
        pl = None
        POLARS_AVAILABLE = False


_NUMERIC_ONLY_AGGREGATIONS = frozenset(
    {
        "Average",
        "LengthWeightedAverage",
        "LengthWeightedPercentile",
        "SumProportionOfData",
        "SumProportionOfTarget",
        "Sum",
        "IndexOfMax",
        "IndexOfMin",
        "Min",
        "Max",
    }
)


def is_polars_available() -> bool:
    """Check if Polars is available for the Polars-native merge backend."""
    return POLARS_AVAILABLE


def _is_numeric_dtype(dtype: Any) -> bool:
    """Return True if the Polars dtype represents a numeric type."""
    try:
        return bool(dtype.is_numeric())
    except AttributeError:
        pass
    numeric_dtypes = (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    )
    return dtype in numeric_dtypes


def _ensure_polars_dataframe(name: str, frame: object) -> pl.DataFrame:
    if pl is None:
        raise ImportError(
            "polars is required for on_slk_intervals_polars. "
            "Install with: pip install pyroads[polars]"
        )
    if isinstance(frame, pl.LazyFrame):
        raise InvalidDataFrameError(
            f"`{name}` parameter is a `polars.LazyFrame`. Please call `.collect()` "
            "before passing it to on_slk_intervals_polars()."
        )
    if not isinstance(frame, pl.DataFrame):
        raise InvalidDataFrameError(
            f"`{name}` parameter must be a polars DataFrame, received {type(frame)}"
        )
    return frame


def _validate_polars_inputs(
    target: object,
    data: object,
    join_left: List[str],
    column_actions: List[Any],
    from_to: Tuple[str, str],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    if not isinstance(join_left, list):
        raise TypeError("`join_left` must be a list of column names.")

    target_df = _ensure_polars_dataframe("target", target)
    data_df = _ensure_polars_dataframe("data", data)

    if len(set(target_df.columns)) != len(target_df.columns):
        raise DuplicateLabelError("`target` dataframe has duplicated column names.")
    if len(set(data_df.columns)) != len(data_df.columns):
        raise DuplicateLabelError("`data` dataframe has duplicated column names.")

    slk_from, slk_to = from_to
    required = [*join_left, slk_from, slk_to]
    missing_messages: List[str] = []
    for column_name in required:
        in_target = column_name in target_df.columns
        in_data = column_name in data_df.columns
        if not in_target and not in_data:
            missing_messages.append(
                f"Column '{column_name}' is missing from both `target` and `data`."
            )
        elif not in_target:
            missing_messages.append(f"Column '{column_name}' is missing from `target`.")
        elif not in_data:
            missing_messages.append(f"Column '{column_name}' is missing from `data`.")
    if missing_messages:
        raise InvalidJoinConfigurationError(
            "Please check the `join_left` and `from_to` parameters. "
            "Specified columns must be present and have matching names in both "
            "`target` and `data`:\n" + "\n".join(missing_messages)
        )

    if target_df.filter(pl.col(slk_from) == pl.col(slk_to)).height > 0:
        raise ZeroLengthSegmentError(
            f"`target` dataframe has rows where {slk_from} == {slk_to}. "
            "The merge tool does not work with zero length segments."
        )
    if data_df.filter(pl.col(slk_from) == pl.col(slk_to)).height > 0:
        raise ZeroLengthSegmentError(
            f"`data` dataframe has rows where {slk_from} == {slk_to}. "
            "The merge tool does not work with zero length segments."
        )

    target_columns = set(target_df.columns)
    for action in column_actions:
        rename = action.rename
        if rename in target_columns:
            if rename == action.column_name:
                raise OutputCollisionError(
                    "Cannot merge column "
                    f"'{action.column_name}' into target because the target already "
                    "contains a column of that name. Please consider using "
                    "the rename parameter; `Action(..., rename='xyz')`."
                )
            raise OutputCollisionError(
                "Cannot merge column "
                f"'{action.column_name}' as '{rename}' into target because the "
                f"target already contains a column named '{rename}'."
            )

    invalid_messages: List[str] = []
    for action in column_actions:
        column_name = action.column_name
        if column_name not in data_df.columns:
            continue
        aggregation_name = action.aggregation.type.name
        if aggregation_name not in _NUMERIC_ONLY_AGGREGATIONS:
            continue
        if not _is_numeric_dtype(data_df.schema[column_name]):
            invalid_messages.append(
                "Aggregation "
                f"'{aggregation_name}' requires numeric data in column '{column_name}' "
                f"(dtype: {data_df.schema[column_name]})."
            )
    if invalid_messages:
        raise InvalidAggregationError("\n".join(invalid_messages))

    return target_df, data_df


def on_slk_intervals_polars(
    target: pl.DataFrame,
    data: pl.DataFrame,
    join_left: List[str],
    column_actions: List[Any],
    from_to: Tuple[str, str],
    verbose: bool = False,
    n_jobs: Optional[int] = None,
) -> pl.DataFrame:
    """Merge and aggregate interval data from Polars DataFrames.

    This is a drop-in Polars equivalent of :func:`on_slk_intervals_numba` and
    returns identical results. All join groups are merged in one native call
    that releases the GIL and runs groups in parallel.

    Args:
        target: Polars DataFrame containing the segments to populate.
        data: Polars DataFrame providing the measurements to aggregate.
        join_left: Ordered list of column names defining grouping keys. Rows
            with a missing key are not matched, as in the pandas backend.
        column_actions: Sequence of :class:`Action` instances describing
            aggregations.
        from_to: Tuple of (start column, end column) names describing each
            interval (half-open, start inclusive, end exclusive).
        verbose: If True, prints diagnostic timing information.
        n_jobs: Number of native worker threads. Defaults to the global Rayon
            pool, sized by ``RAYON_NUM_THREADS`` or the number of CPUs.

    Returns:
        A new Polars DataFrame with the same rows as ``target`` plus one
        column per entry in ``column_actions``.
    """
    if not NUMBA_AVAILABLE:
        raise ImportError(
            "Numba is required for on_slk_intervals_polars. "
            "Install with: pip install pyroads"
        )
    if n_jobs is not None and n_jobs < 1:
        raise ValueError("`n_jobs` must be a positive integer.")

    from . import _grouped

    start_time = time.perf_counter()
    slk_from, slk_to = from_to

    target_df, data_df = _validate_polars_inputs(
        target, data, join_left, column_actions, from_to
    )

    categorical_actions = []
    numeric_actions = []
    for action in column_actions:
        column_dtype = data_df.schema.get(action.column_name)
        if (
            action.aggregation.type.value == AGG_KEEP_LONGEST
            and column_dtype is not None
            and not _is_numeric_dtype(column_dtype)
        ):
            categorical_actions.append(action)
        else:
            numeric_actions.append(action)

    target_ids, data_ids = _grouped.polars_group_ids(target_df, data_df, join_left)
    total_groups = int(np.count_nonzero(np.bincount(target_ids[target_ids >= 0])))
    if verbose:
        print(
            f"[pyroads.merge] Polars merge: {len(column_actions)} action(s), "
            f"{total_groups} group(s)."
        )

    def as_float(column: str) -> np.ndarray:
        return data_df[column].to_numpy().astype(np.float64, copy=False)

    factorized = {
        column_name: _grouped.factorize(data_df[column_name].to_list())
        for column_name in dict.fromkeys(action.column_name for action in categorical_actions)
    }
    agg_codes = [_get_agg_type_code(action.aggregation) for action in numeric_actions]
    numeric_results, categorical_results = _grouped.merge_groups(
        target_ids,
        target_df[slk_from].to_numpy().astype(np.float64, copy=False),
        target_df[slk_to].to_numpy().astype(np.float64, copy=False),
        data_ids,
        as_float(slk_from),
        as_float(slk_to),
        np.arange(data_df.height, dtype=np.int64),
        numeric_values=[as_float(action.column_name) for action in numeric_actions],
        agg_types=[agg_type for agg_type, _ in agg_codes],
        percentiles=[percentile for _, percentile in agg_codes],
        category_codes=[factorized[action.column_name][0] for action in categorical_actions],
        n_threads=n_jobs or 0,
    )

    output_columns: Dict[str, pl.Series] = {}
    for action, values in zip(numeric_actions, numeric_results):
        output_columns[action.rename] = pl.Series(action.rename, values, dtype=pl.Float64)
    for action, codes in zip(categorical_actions, categorical_results):
        values = _grouped.decode(codes, factorized[action.column_name][1])
        output_columns[action.rename] = pl.Series(action.rename, values.tolist())
    result = target_df.with_columns(
        [output_columns[action.rename] for action in column_actions]
    )

    elapsed = time.perf_counter() - start_time
    if verbose:
        print(f"[pyroads.merge] Polars merge completed in {elapsed:.2f}s")

    try:
        from . import merge as merge_module

        if hasattr(merge_module, "_emit_performance_event"):
            merge_module._emit_performance_event(
                "on_slk_intervals_polars",
                duration=elapsed,
                groups=float(total_groups),
                actions=float(len(column_actions)),
                rows=float(target_df.height),
            )
    except Exception:
        pass  # Performance logging is optional

    return result
