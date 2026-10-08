"""Backend-agnostic grouped interval merge.

Both the pandas and Polars entry points reduce their inputs to flat NumPy
arrays plus a shared integer group id per row. Rows are stably sorted by group
id so every group occupies a contiguous slice, and all groups are then merged
in a single native call that releases the GIL and parallelises across groups.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import _numba_merge as kernels


def pandas_group_ids(
    target: pd.DataFrame, data: pd.DataFrame, join_left: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Label rows of both frames with shared group ids; missing keys get -1."""
    keys = pd.concat([target[join_left], data[join_left]], ignore_index=True)
    ids = keys.groupby(join_left, sort=False, dropna=True).ngroup()
    ids = ids.fillna(-1).to_numpy(dtype=np.int64)
    return ids[: len(target)], ids[len(target) :]


def polars_group_ids(target: Any, data: Any, join_left: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Polars equivalent of :func:`pandas_group_ids`."""
    import polars as pl

    try:
        keys = pl.concat(
            [target.select(join_left), data.select(join_left)], how="vertical_relaxed"
        )
    except (pl.exceptions.PolarsError, TypeError):
        # Key dtypes with no common supertype can never match.
        return (
            np.full(target.height, -1, dtype=np.int64),
            np.full(data.height, -1, dtype=np.int64),
        )
    has_missing_key = pl.any_horizontal([pl.col(column).is_null() for column in join_left])
    group_id = (
        pl.when(has_missing_key)
        .then(pl.lit(-1, dtype=pl.Int64))
        .otherwise(pl.struct(join_left).rank("dense").cast(pl.Int64) - 1)
    )
    ids = keys.select(group_id.alias("id")).to_series().to_numpy().astype(np.int64, copy=False)
    return ids[: target.height], ids[target.height :]


def factorize(values: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return int64 codes (-1 for missing) and the matching unique values."""
    codes, uniques = pd.factorize(values, sort=False, use_na_sentinel=True)
    return codes.astype(np.int64, copy=False), np.asarray(uniques, dtype=object)


def decode(codes: np.ndarray, uniques: np.ndarray) -> np.ndarray:
    """Map codes back to their values, with ``None`` for -1."""
    result = np.full(len(codes), None, dtype=object)
    valid = codes >= 0
    result[valid] = uniques[codes[valid]]
    return result


def _sort_by_group(ids: np.ndarray, n_groups: int) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(ids, kind="stable")
    order = order[np.searchsorted(ids[order], 0) :]
    offsets = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(np.bincount(ids[order], minlength=n_groups), out=offsets[1:])
    return order, offsets


def _stack(columns: Sequence[np.ndarray], order: np.ndarray, dtype: Any) -> np.ndarray:
    if not columns:
        return np.empty((0, len(order)), dtype=dtype)
    return np.ascontiguousarray(np.vstack([column[order] for column in columns]), dtype=dtype)


def merge_groups(
    target_ids: np.ndarray,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
    data_ids: np.ndarray,
    data_starts: np.ndarray,
    data_ends: np.ndarray,
    original_indices: np.ndarray,
    numeric_values: Sequence[np.ndarray],
    agg_types: Sequence[int],
    percentiles: Sequence[float],
    category_codes: Sequence[np.ndarray] = (),
    n_threads: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge every group and return results in the original target row order.

    Returns a ``(n_numeric, n_target)`` float64 array and a
    ``(n_categorical, n_target)`` int64 array of category codes (-1 if none).
    """
    n_target = len(target_ids)
    numeric_out = np.full((len(numeric_values), n_target), np.nan, dtype=np.float64)
    categorical_out = np.full((len(category_codes), n_target), -1, dtype=np.int64)
    n_groups = int(max(target_ids.max(initial=-1), data_ids.max(initial=-1))) + 1
    if n_groups == 0:
        return numeric_out, categorical_out

    target_order, target_offsets = _sort_by_group(target_ids, n_groups)
    data_order, data_offsets = _sort_by_group(data_ids, n_groups)
    sorted_target_starts = np.ascontiguousarray(target_starts[target_order], dtype=np.float64)
    sorted_target_ends = np.ascontiguousarray(target_ends[target_order], dtype=np.float64)
    sorted_data_starts = np.ascontiguousarray(data_starts[data_order], dtype=np.float64)
    sorted_data_ends = np.ascontiguousarray(data_ends[data_order], dtype=np.float64)
    sorted_original = np.ascontiguousarray(original_indices[data_order], dtype=np.int64)
    numeric_matrix = _stack(numeric_values, data_order, np.float64)
    category_matrix = _stack(category_codes, data_order, np.int64)
    agg_type_array = np.asarray(agg_types, dtype=np.int64)
    percentile_array = np.asarray(percentiles, dtype=np.float64)

    native_merge = getattr(kernels._rust_native, "merge_groups", None)
    if native_merge is not None:
        numeric, categorical = native_merge(
            sorted_target_starts,
            sorted_target_ends,
            target_offsets,
            sorted_data_starts,
            sorted_data_ends,
            data_offsets,
            sorted_original,
            numeric_matrix,
            agg_type_array,
            percentile_array,
            category_matrix,
            n_threads,
        )
        numeric_rows = np.asarray(numeric).reshape(len(target_order), len(numeric_values)).T
        categorical_rows = np.asarray(categorical).reshape(len(target_order), len(category_codes)).T
    else:
        numeric_rows, categorical_rows = _merge_groups_python(
            sorted_target_starts,
            sorted_target_ends,
            target_offsets,
            sorted_data_starts,
            sorted_data_ends,
            data_offsets,
            sorted_original,
            numeric_matrix,
            agg_type_array,
            percentile_array,
            category_matrix,
        )
    numeric_out[:, target_order] = numeric_rows
    categorical_out[:, target_order] = categorical_rows
    return numeric_out, categorical_out


def _merge_groups_python(
    target_starts: np.ndarray,
    target_ends: np.ndarray,
    target_offsets: np.ndarray,
    data_starts: np.ndarray,
    data_ends: np.ndarray,
    data_offsets: np.ndarray,
    original_indices: np.ndarray,
    numeric_matrix: np.ndarray,
    agg_types: np.ndarray,
    percentiles: np.ndarray,
    category_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-group fallback used when the native ``merge_groups`` kernel is unavailable."""
    numeric_rows = np.full((len(numeric_matrix), len(target_starts)), np.nan, dtype=np.float64)
    categorical_rows = np.full((len(category_matrix), len(target_starts)), -1, dtype=np.int64)
    data_lengths = data_ends - data_starts
    target_lengths = target_ends - target_starts
    for group in range(len(target_offsets) - 1):
        target_first, target_last = int(target_offsets[group]), int(target_offsets[group + 1])
        data_first, data_last = int(data_offsets[group]), int(data_offsets[group + 1])
        if target_first == target_last or data_first == data_last:
            continue
        targets = slice(target_first, target_last)
        rows = slice(data_first, data_last)
        tgt_idx, data_idx, overlap_lens = kernels._find_overlapping_intervals_sorted(
            target_starts[targets], target_ends[targets], data_starts[rows], data_ends[rows]
        )
        if len(tgt_idx) == 0:
            continue
        n_targets = target_last - target_first
        for action, values in enumerate(numeric_matrix):
            numeric_rows[action, targets] = kernels._aggregate_all_targets_numeric(
                n_targets,
                tgt_idx,
                data_idx,
                overlap_lens,
                values[rows],
                data_lengths[rows],
                original_indices[rows],
                target_lengths[targets],
                int(agg_types[action]),
                float(percentiles[action]),
            )
        for action, codes in enumerate(category_matrix):
            group_codes = codes[rows].astype(object)
            group_codes[codes[rows] < 0] = None
            best = kernels._aggregate_keep_longest_categorical(
                n_targets, tgt_idx, data_idx, overlap_lens, group_codes
            )
            categorical_rows[action, targets] = [-1 if code is None else code for code in best]
    return numeric_rows, categorical_rows
