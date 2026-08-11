"""Polars-backed interval merge implementation.

This module reuses the Numba sparse-overlap kernels from :mod:`_numba_merge`
so that numeric aggregation results are identical to the pandas/Numba path.
Only the DataFrame-shaped glue (grouping, column extraction, result assembly)
is Polars-native. Groups are independent of one another, so they are
processed concurrently via a thread pool; the core Numba kernels are compiled
with ``nogil=True`` so this achieves genuine multi-core parallelism instead of
serializing on the GIL.

Requires: polars, numba (numba is already a core dependency of the package).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
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
from ._numba_merge import (
    AGG_KEEP_LONGEST,
    NUMBA_AVAILABLE,
    _aggregate_all_targets_numeric,
    _aggregate_keep_longest_categorical,
    _find_overlapping_intervals_sorted,
    _get_agg_type_code,
)

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

_ROW_IDX_COL = "__pyroads.merge_target_row_idx__"
_DATA_IDX_COL = "__pyroads.merge_data_row_idx__"
_SEGMENT_LEN_COL = "__pyroads.merge_segment_len__"


def is_polars_available() -> bool:
    """Check if Polars is available for the Polars-native merge backend."""
    return POLARS_AVAILABLE


def _normalize_group_key(key: Any) -> tuple:
    """Return a deterministic tuple key for a Polars ``partition_by`` group."""
    if isinstance(key, tuple):
        return key
    if isinstance(key, list):
        return tuple(key)
    return (key,)


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
    """Merge and aggregate interval data using a Polars-native, multithreaded path.

    This is a drop-in Polars equivalent of :func:`on_slk_intervals_numba`. It
    partitions ``target`` and ``data`` by ``join_left`` using Polars, then
    processes each independent group concurrently in a thread pool. The
    numeric aggregation for each group calls the same Numba kernels used by
    the pandas/Numba backend, guaranteeing identical results.

    Args:
        target: Polars DataFrame containing the segments to populate.
        data: Polars DataFrame providing the measurements to aggregate.
        join_left: Ordered list of column names defining grouping keys.
        column_actions: Sequence of :class:`Action` instances describing
            aggregations.
        from_to: Tuple of (start column, end column) names describing each
            interval (half-open, start inclusive, end exclusive).
        verbose: If True, prints diagnostic timing information.
        n_jobs: Number of worker threads to use. Defaults to the number of
            available CPUs (capped at 32).

    Returns:
        A new Polars DataFrame with the same rows as ``target`` plus one
        column per entry in ``column_actions``.
    """
    if not NUMBA_AVAILABLE:
        raise ImportError(
            "Numba is required for on_slk_intervals_polars. "
            "Install with: pip install pyroads"
        )

    start_time = time.perf_counter()
    slk_from, slk_to = from_to

    target_df, data_df = _validate_polars_inputs(
        target, data, join_left, column_actions, from_to
    )

    n_target = target_df.height

    target_indexed = target_df.with_row_index(_ROW_IDX_COL)

    data_needed_cols = list({action.column_name for action in column_actions})
    data_subset = data_df.select([*join_left, slk_from, slk_to, *data_needed_cols])
    data_subset = data_subset.with_row_index(_DATA_IDX_COL)
    data_subset = data_subset.with_columns(
        (pl.col(slk_to) - pl.col(slk_from)).alias(_SEGMENT_LEN_COL)
    )

    target_groups = target_indexed.partition_by(
        join_left, as_dict=True, maintain_order=True
    )
    data_groups = data_subset.partition_by(join_left, as_dict=True, maintain_order=True)
    data_groups_normalized = {
        _normalize_group_key(key): group for key, group in data_groups.items()
    }

    # Determine output dtype per action up-front (numeric float64, or object
    # for KeepLongest on non-numeric columns).
    is_categorical: Dict[str, bool] = {}
    output_arrays: Dict[str, np.ndarray] = {}
    for action in column_actions:
        agg_type_value = action.aggregation.type.value
        column_dtype = data_df.schema.get(action.column_name)
        if (
            agg_type_value == AGG_KEEP_LONGEST
            and column_dtype is not None
            and not _is_numeric_dtype(column_dtype)
        ):
            is_categorical[action.rename] = True
            output_arrays[action.rename] = np.full(n_target, None, dtype=object)
        else:
            is_categorical[action.rename] = False
            output_arrays[action.rename] = np.full(n_target, np.nan, dtype=np.float64)

    def _process_group(
        key: Any, tgt_group: pl.DataFrame
    ) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        data_group = data_groups_normalized.get(_normalize_group_key(key))
        if data_group is None or data_group.height == 0:
            return None

        tgt_starts = tgt_group[slk_from].to_numpy().astype(np.float64)
        tgt_ends = tgt_group[slk_to].to_numpy().astype(np.float64)
        tgt_lengths = tgt_ends - tgt_starts
        tgt_row_positions = tgt_group[_ROW_IDX_COL].to_numpy().astype(np.int64)

        data_starts = data_group[slk_from].to_numpy().astype(np.float64)
        data_ends = data_group[slk_to].to_numpy().astype(np.float64)
        data_lengths = data_group[_SEGMENT_LEN_COL].to_numpy().astype(np.float64)
        original_indices = data_group[_DATA_IDX_COL].to_numpy().astype(np.int64)

        tgt_idx, data_idx, overlap_lens = _find_overlapping_intervals_sorted(
            tgt_starts, tgt_ends, data_starts, data_ends
        )
        if len(tgt_idx) == 0:
            return None

        group_results: Dict[str, np.ndarray] = {}
        for action in column_actions:
            rename = action.rename
            agg_type, percentile = _get_agg_type_code(action.aggregation)

            if is_categorical[rename]:
                col_values = np.array(
                    data_group[action.column_name].to_list(), dtype=object
                )
                agg_results = _aggregate_keep_longest_categorical(
                    n_targets=tgt_group.height,
                    tgt_indices=tgt_idx,
                    data_indices=data_idx,
                    overlap_lens=overlap_lens,
                    col_values=col_values,
                )
            else:
                col_values = (
                    data_group[action.column_name].to_numpy().astype(np.float64)
                )
                agg_results = _aggregate_all_targets_numeric(
                    n_targets=tgt_group.height,
                    tgt_indices=tgt_idx,
                    data_indices=data_idx,
                    overlap_lens=overlap_lens,
                    col_values=col_values,
                    data_lengths=data_lengths,
                    original_indices=original_indices,
                    target_lengths=tgt_lengths,
                    agg_type=agg_type,
                    percentile=percentile,
                )
            group_results[rename] = agg_results

        return tgt_row_positions, group_results

    group_items = list(target_groups.items())
    max_workers = n_jobs or min(32, (os.cpu_count() or 1))

    if verbose:
        print(
            f"[pyroads.merge] Polars merge: {len(column_actions)} action(s), "
            f"{len(group_items)} group(s), {max_workers} worker thread(s)."
        )

    if max_workers <= 1 or len(group_items) <= 1:
        for key, tgt_group in group_items:
            outcome = _process_group(key, tgt_group)
            if outcome is None:
                continue
            positions, group_results = outcome
            for rename, values in group_results.items():
                output_arrays[rename][positions] = values
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_process_group, key, tgt_group)
                for key, tgt_group in group_items
            ]
            for future in futures:
                outcome = future.result()
                if outcome is None:
                    continue
                positions, group_results = outcome
                for rename, values in group_results.items():
                    output_arrays[rename][positions] = values

    result = target_df.clone()
    for action in column_actions:
        rename = action.rename
        values = output_arrays[rename]
        if is_categorical[rename]:
            result = result.with_columns(pl.Series(rename, values.tolist()))
        else:
            result = result.with_columns(pl.Series(rename, values, dtype=pl.Float64))

    elapsed = time.perf_counter() - start_time
    if verbose:
        print(f"[pyroads.merge] Polars merge completed in {elapsed:.2f}s")

    try:
        from . import merge as merge_module

        if hasattr(merge_module, "_emit_performance_event"):
            merge_module._emit_performance_event(
                "on_slk_intervals_polars",
                duration=elapsed,
                groups=float(len(group_items)),
                actions=float(len(column_actions)),
                rows=float(n_target),
            )
    except Exception:
        pass  # Performance logging is optional

    return result
