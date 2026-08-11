"""
Numba-accelerated sparse interval merge implementation.

This module provides memory-efficient interval merging using a sweep-line
algorithm that avoids creating dense overlap matrices. Combined with Numba
JIT compilation, this approach is 20-100x faster than the legacy implementation
and can handle datasets that would cause OOM errors with the dense matrix approach.

Memory complexity: O(actual_overlaps) instead of O(n_target × n_data)
Time complexity: sparse-overlap workloads are much faster than the dense matrix
path, but worst-case scanning can still approach O(n_target × n_data)

Requires: numba>=0.57
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from . import _validation as validation

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Stub decorator for when numba is not installed
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range


# =============================================================================
# CORE NUMBA-ACCELERATED INTERVAL INTERSECTION
# =============================================================================


@njit(cache=True, nogil=True)
def _find_overlapping_intervals_sorted(
    tgt_starts: np.ndarray,
    tgt_ends: np.ndarray,
    data_starts: np.ndarray,
    data_ends: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find all overlapping intervals using a sorted sweep approach.

    This is a memory-efficient algorithm that only stores actual overlaps,
    avoiding the O(n_target × n_data) dense matrix.

    Args:
        tgt_starts: Target interval start positions (float64)
        tgt_ends: Target interval end positions (float64)
        data_starts: Data interval start positions (float64)
        data_ends: Data interval end positions (float64)

    Returns:
        Tuple of (target_indices, data_indices, overlap_lengths) as numpy arrays.
        Only contains entries where overlap > 0.
    """
    n_tgt = len(tgt_starts)
    n_data = len(data_starts)

    # Sort data by END position for efficient binary search
    # This allows us to skip data segments that definitely don't overlap
    data_order = np.argsort(data_ends)
    sorted_data_ends = data_ends[data_order]
    sorted_data_starts = data_starts[data_order]

    # Pre-allocate result arrays with reasonable initial capacity
    # Estimate average overlaps per target segment
    initial_capacity = max(n_tgt * 5, 1000)
    tgt_indices = np.empty(initial_capacity, dtype=np.int64)
    data_indices = np.empty(initial_capacity, dtype=np.int64)
    overlap_lens = np.empty(initial_capacity, dtype=np.float64)
    result_count = 0

    for t in range(n_tgt):
        t_start = tgt_starts[t]
        t_end = tgt_ends[t]

        if t_end <= t_start:
            continue  # Skip zero or negative length targets

        # Binary search to find first data segment that could overlap
        # Data segment overlaps if: data_end > t_start AND data_start < t_end
        # Since data is sorted by END, find first where data_end > t_start
        lo, hi = 0, n_data
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_data_ends[mid] <= t_start:
                lo = mid + 1
            else:
                hi = mid

        # Scan forward through potentially overlapping data segments
        # Since sorted by END, once we find data_end > t_start, we scan all remaining
        for d_sorted in range(lo, n_data):
            d_start = sorted_data_starts[d_sorted]
            d_end = sorted_data_ends[d_sorted]

            # Check if this segment overlaps
            # Overlap condition: data_start < t_end (data_end > t_start is guaranteed by binary search)
            if d_start >= t_end:
                # This segment doesn't overlap, but continue - later segments might
                # (since sorted by END, not START)
                continue

            # Calculate overlap
            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap_len = overlap_end - overlap_start

            if overlap_len > 0:
                # Resize arrays if needed
                if result_count >= len(tgt_indices):
                    new_capacity = len(tgt_indices) * 2
                    new_tgt = np.empty(new_capacity, dtype=np.int64)
                    new_data = np.empty(new_capacity, dtype=np.int64)
                    new_lens = np.empty(new_capacity, dtype=np.float64)
                    new_tgt[:result_count] = tgt_indices[:result_count]
                    new_data[:result_count] = data_indices[:result_count]
                    new_lens[:result_count] = overlap_lens[:result_count]
                    tgt_indices = new_tgt
                    data_indices = new_data
                    overlap_lens = new_lens

                # Store the overlap (map back to original data index)
                tgt_indices[result_count] = t
                data_indices[result_count] = data_order[d_sorted]
                overlap_lens[result_count] = overlap_len
                result_count += 1

    return (
        tgt_indices[:result_count],
        data_indices[:result_count],
        overlap_lens[:result_count],
    )


# =============================================================================
# NUMBA-ACCELERATED AGGREGATION FUNCTIONS
# =============================================================================


@njit(cache=True)
def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted average, ignoring NaN values."""
    total_weight = 0.0
    weighted_sum = 0.0
    for i in range(len(values)):
        v = values[i]
        w = weights[i]
        if not np.isnan(v) and w > 0:
            weighted_sum += v * w
            total_weight += w
    if total_weight == 0.0:
        return np.nan
    return weighted_sum / total_weight


@njit(cache=True)
def _simple_average(values: np.ndarray) -> float:
    """Compute simple average, ignoring NaN values."""
    total = 0.0
    count = 0
    for i in range(len(values)):
        v = values[i]
        if not np.isnan(v):
            total += v
            count += 1
    if count == 0:
        return np.nan
    return total / count


@njit(cache=True)
def _first_valid(values: np.ndarray, orig_indices: np.ndarray) -> float:
    """Return value with smallest original index (first in data order)."""
    min_idx = np.iinfo(np.int64).max
    first_val = np.nan
    for i in range(len(values)):
        v = values[i]
        idx = orig_indices[i]
        if not np.isnan(v) and idx < min_idx:
            min_idx = idx
            first_val = v
    return first_val


@njit(cache=True)
def _sum_values(values: np.ndarray) -> float:
    """Sum all non-NaN values."""
    total = 0.0
    has_value = False
    for i in range(len(values)):
        v = values[i]
        if not np.isnan(v):
            total += v
            has_value = True
    return total if has_value else np.nan


@njit(cache=True)
def _max_value(values: np.ndarray) -> float:
    """Find maximum non-NaN value."""
    result = -np.inf
    has_value = False
    for i in range(len(values)):
        v = values[i]
        if not np.isnan(v):
            if v > result:
                result = v
            has_value = True
    return result if has_value else np.nan


@njit(cache=True)
def _min_value(values: np.ndarray) -> float:
    """Find minimum non-NaN value."""
    result = np.inf
    has_value = False
    for i in range(len(values)):
        v = values[i]
        if not np.isnan(v):
            if v < result:
                result = v
            has_value = True
    return result if has_value else np.nan


@njit(cache=True)
def _argmax_value(values: np.ndarray, indices: np.ndarray) -> float:
    """Return original index of maximum value."""
    max_val = -np.inf
    max_idx = -1
    for i in range(len(values)):
        v = values[i]
        if not np.isnan(v) and v > max_val:
            max_val = v
            max_idx = indices[i]
    return float(max_idx) if max_idx >= 0 else np.nan


@njit(cache=True)
def _argmin_value(values: np.ndarray, indices: np.ndarray) -> float:
    """Return original index of minimum value."""
    min_val = np.inf
    min_idx = -1
    for i in range(len(values)):
        v = values[i]
        if not np.isnan(v) and v < min_val:
            min_val = v
            min_idx = indices[i]
    return float(min_idx) if min_idx >= 0 else np.nan


@njit(cache=True)
def _keep_longest_segment(values: np.ndarray, overlaps: np.ndarray) -> float:
    """Return value with maximum overlap length."""
    max_overlap = -1.0
    result = np.nan
    for i in range(len(values)):
        v = values[i]
        o = overlaps[i]
        if not np.isnan(v) and o > max_overlap:
            max_overlap = o
            result = v
    return result


@njit(cache=True)
def _sum_proportion_of_data(
    values: np.ndarray, overlaps: np.ndarray, data_lengths: np.ndarray
) -> float:
    """Sum of (value × overlap / data_length)."""
    total = 0.0
    has_value = False
    for i in range(len(values)):
        v = values[i]
        o = overlaps[i]
        d_len = data_lengths[i]
        if not np.isnan(v) and d_len > 0:
            total += v * o / d_len
            has_value = True
    return total if has_value else np.nan


@njit(cache=True)
def _sum_proportion_of_target(
    values: np.ndarray, overlaps: np.ndarray, target_length: float
) -> float:
    """Sum of (value × overlap) / target_length."""
    if target_length <= 0:
        return np.nan
    total = 0.0
    has_value = False
    for i in range(len(values)):
        v = values[i]
        o = overlaps[i]
        if not np.isnan(v):
            total += v * o
            has_value = True
    return (total / target_length) if has_value else np.nan


@njit(cache=True)
def _length_weighted_percentile(
    values: np.ndarray, overlaps: np.ndarray, percentile: float
) -> float:
    """
    Compute length-weighted percentile.

    Sorts values by value, then interpolates based on cumulative overlap lengths.
    """
    # Filter out NaN values
    valid_count = 0
    for i in range(len(values)):
        if not np.isnan(values[i]) and overlaps[i] > 0:
            valid_count += 1

    if valid_count == 0:
        return np.nan

    valid_values = np.empty(valid_count, dtype=np.float64)
    valid_overlaps = np.empty(valid_count, dtype=np.float64)
    idx = 0
    for i in range(len(values)):
        if not np.isnan(values[i]) and overlaps[i] > 0:
            valid_values[idx] = values[i]
            valid_overlaps[idx] = overlaps[i]
            idx += 1

    if valid_count == 1:
        return valid_values[0]

    # Sort by value
    order = np.argsort(valid_values)
    sorted_vals = valid_values[order]
    sorted_weights = valid_overlaps[order]

    # Compute cumulative midpoint weights
    # x_coords[i] = sum of weights up to midpoint of segment i
    x_coords = np.zeros(valid_count, dtype=np.float64)
    cumsum = 0.0
    for i in range(valid_count):
        if i == 0:
            x_coords[i] = 0.0
        else:
            # Midpoint between previous and current
            pair_avg = (sorted_weights[i - 1] + sorted_weights[i]) * 0.5
            cumsum += pair_avg
            x_coords[i] = cumsum

    total = x_coords[-1]
    if total <= 0:
        return sorted_vals[-1]

    # Normalize to [0, 1]
    for i in range(valid_count):
        x_coords[i] /= total

    # Linear interpolation
    target_x = percentile

    # Find bracketing indices
    if target_x <= x_coords[0]:
        return sorted_vals[0]
    if target_x >= x_coords[-1]:
        return sorted_vals[-1]

    for i in range(1, valid_count):
        if x_coords[i] >= target_x:
            # Interpolate between i-1 and i
            x0, x1 = x_coords[i - 1], x_coords[i]
            y0, y1 = sorted_vals[i - 1], sorted_vals[i]
            if x1 == x0:
                return y0
            t = (target_x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    return sorted_vals[-1]


# =============================================================================
# AGGREGATION TYPE ENUM MAPPING
# =============================================================================

# Aggregation type codes (must match AggregationType enum values)
AGG_AVERAGE = 3
AGG_LENGTH_WEIGHTED_AVERAGE = 4
AGG_LENGTH_WEIGHTED_PERCENTILE = 5
AGG_FIRST = 6
AGG_SUM_PROPORTION_OF_DATA = 7
AGG_SUM_PROPORTION_OF_TARGET = 8
AGG_SUM = 9
AGG_INDEX_OF_MAX = 10
AGG_INDEX_OF_MIN = 11
AGG_MIN = 12
AGG_MAX = 13
AGG_KEEP_LONGEST_SEGMENT = 1
AGG_KEEP_LONGEST = 2


# =============================================================================
# MAIN SPARSE AGGREGATION ENGINE
# =============================================================================


@njit(cache=True)
def _aggregate_single_target(
    values: np.ndarray,
    overlaps: np.ndarray,
    data_lengths: np.ndarray,
    original_indices: np.ndarray,
    target_length: float,
    agg_type: int,
    percentile: float,
) -> float:
    """
    Aggregate values for a single target using the specified aggregation type.

    Args:
        values: Data values that overlap with target
        overlaps: Overlap lengths for each value
        data_lengths: Original lengths of data segments
        original_indices: Original indices of data rows
        target_length: Length of target segment
        agg_type: Aggregation type code
        percentile: Percentile value (only used for LengthWeightedPercentile)

    Returns:
        Aggregated value
    """
    if len(values) == 0:
        return np.nan

    if agg_type == AGG_AVERAGE:
        return _simple_average(values)

    elif agg_type == AGG_LENGTH_WEIGHTED_AVERAGE:
        return _weighted_average(values, overlaps)

    elif agg_type == AGG_LENGTH_WEIGHTED_PERCENTILE:
        return _length_weighted_percentile(values, overlaps, percentile)

    elif agg_type == AGG_FIRST:
        return _first_valid(values, original_indices)

    elif agg_type == AGG_SUM:
        return _sum_values(values)

    elif agg_type == AGG_SUM_PROPORTION_OF_DATA:
        return _sum_proportion_of_data(values, overlaps, data_lengths)

    elif agg_type == AGG_SUM_PROPORTION_OF_TARGET:
        return _sum_proportion_of_target(values, overlaps, target_length)

    elif agg_type == AGG_MAX:
        return _max_value(values)

    elif agg_type == AGG_MIN:
        return _min_value(values)

    elif agg_type == AGG_INDEX_OF_MAX:
        return _argmax_value(values, original_indices)

    elif agg_type == AGG_INDEX_OF_MIN:
        return _argmin_value(values, original_indices)

    elif agg_type == AGG_KEEP_LONGEST_SEGMENT:
        return _keep_longest_segment(values, overlaps)

    elif agg_type == AGG_KEEP_LONGEST:
        # For numeric values, same as keep_longest_segment
        # Categorical handling done in Python layer
        return _keep_longest_segment(values, overlaps)

    return np.nan


@njit(cache=True, nogil=True)
def _aggregate_all_targets_numeric(
    n_targets: int,
    tgt_indices: np.ndarray,
    data_indices: np.ndarray,
    overlap_lens: np.ndarray,
    col_values: np.ndarray,
    data_lengths: np.ndarray,
    original_indices: np.ndarray,
    target_lengths: np.ndarray,
    agg_type: int,
    percentile: float,
) -> np.ndarray:
    """
    Aggregate all targets for a single numeric column.

    Note: this is called once per (group, action) pair -- often thousands of
    times per merge, each call covering only a small number of targets. Numba's
    ``parallel=True`` (prange) was benchmarked here and found to be a net loss:
    spinning up its internal thread team on every one of these small calls costs
    more in synchronization overhead than the parallel loop saves (~3x slower,
    2.9M-row benchmark: 12.2s vs 4.5s without it). ``nogil=True`` is kept
    because it enables real multi-core speedup when this function is called
    concurrently from multiple Python threads, as the Polars backend does.

    Args:
        n_targets: Number of target rows
        tgt_indices: Target indices from sparse overlap
        data_indices: Data indices from sparse overlap
        overlap_lens: Overlap lengths from sparse overlap
        col_values: Column values from data
        data_lengths: Lengths of data segments
        original_indices: Original row indices from data
        target_lengths: Lengths of target segments
        agg_type: Aggregation type code
        percentile: Percentile value (for LengthWeightedPercentile)

    Returns:
        Array of aggregated values, one per target
    """
    results = np.full(n_targets, np.nan, dtype=np.float64)

    if len(tgt_indices) == 0:
        return results

    # Build grouped structure: count overlaps per target
    counts = np.zeros(n_targets, dtype=np.int64)
    for i in range(len(tgt_indices)):
        counts[tgt_indices[i]] += 1

    # Compute offsets for each target's data
    offsets = np.zeros(n_targets + 1, dtype=np.int64)
    for t in range(n_targets):
        offsets[t + 1] = offsets[t] + counts[t]

    # Build grouped arrays
    grouped_data_idx = np.empty(len(tgt_indices), dtype=np.int64)
    grouped_overlaps = np.empty(len(tgt_indices), dtype=np.float64)
    current_pos = np.zeros(n_targets, dtype=np.int64)
    for t in range(n_targets):
        current_pos[t] = offsets[t]

    for i in range(len(tgt_indices)):
        t = tgt_indices[i]
        pos = current_pos[t]
        grouped_data_idx[pos] = data_indices[i]
        grouped_overlaps[pos] = overlap_lens[i]
        current_pos[t] += 1

    # Sequential aggregation across targets (see note above re: parallel=True)
    for t in range(n_targets):
        start = offsets[t]
        end = offsets[t + 1]

        if start == end:
            continue  # No overlaps for this target

        n_overlaps = end - start
        vals = np.empty(n_overlaps, dtype=np.float64)
        ovlps = np.empty(n_overlaps, dtype=np.float64)
        d_lens = np.empty(n_overlaps, dtype=np.float64)
        orig_idx = np.empty(n_overlaps, dtype=np.int64)

        for i in range(n_overlaps):
            d_idx = grouped_data_idx[start + i]
            vals[i] = col_values[d_idx]
            ovlps[i] = grouped_overlaps[start + i]
            d_lens[i] = data_lengths[d_idx]
            orig_idx[i] = original_indices[d_idx]

        results[t] = _aggregate_single_target(
            vals,
            ovlps,
            d_lens,
            orig_idx,
            target_lengths[t],
            agg_type,
            percentile,
        )

    return results


# =============================================================================
# CATEGORICAL AGGREGATION (KeepLongest for non-numeric)
# =============================================================================


def _aggregate_keep_longest_categorical(
    n_targets: int,
    tgt_indices: np.ndarray,
    data_indices: np.ndarray,
    overlap_lens: np.ndarray,
    col_values: np.ndarray,  # Object array
) -> np.ndarray:
    """
    Aggregate KeepLongest for categorical (non-numeric) values.

    This function runs in Python (not Numba) to handle arbitrary Python objects.
    """
    results = np.empty(n_targets, dtype=object)
    results[:] = None

    if len(tgt_indices) == 0:
        return results

    # Group by target
    from collections import defaultdict

    target_data: Dict[int, List[Tuple[Any, float]]] = defaultdict(list)
    for i in range(len(tgt_indices)):
        t = tgt_indices[i]
        d = data_indices[i]
        overlap = overlap_lens[i]
        value = col_values[d]
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            target_data[t].append((value, overlap))

    for t, data_list in target_data.items():
        if not data_list:
            continue

        # Sum overlaps per unique value, track first occurrence
        totals: Dict[Any, float] = {}
        first_idx: Dict[Any, int] = {}

        for idx, (val, overlap) in enumerate(data_list):
            if val not in totals:
                totals[val] = 0.0
                first_idx[val] = idx
            totals[val] += overlap

        # Find value with maximum total overlap (tie-break by first occurrence)
        best_value = None
        best_total = -1.0
        best_order = len(data_list)

        for val, total in totals.items():
            order = first_idx[val]
            if total > best_total or (
                np.isclose(total, best_total) and order < best_order
            ):
                best_total = total
                best_value = val
                best_order = order

        results[t] = best_value

    return results


# =============================================================================
# HIGH-LEVEL PANDAS API
# =============================================================================


def _get_agg_type_code(aggregation) -> Tuple[int, float]:
    """Convert Aggregation object to (type_code, percentile) tuple."""
    agg_type = aggregation.type.value
    percentile = aggregation.percentile if aggregation.percentile is not None else 0.0
    return agg_type, percentile


def _is_numeric_column(series: pd.Series) -> bool:
    """Check if series contains numeric data."""
    return pd.api.types.is_numeric_dtype(series)


def on_slk_intervals_numba(
    target: pd.DataFrame,
    data: pd.DataFrame,
    join_left: List[str],
    column_actions: List[Any],
    from_to: Tuple[str, str],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Memory-efficient, Numba-accelerated interval merge.

    This is a drop-in replacement for on_slk_intervals_optimized() that uses
    a sparse interval intersection algorithm to avoid creating dense matrices.

    Key advantages:
    - Memory: O(actual_overlaps) instead of O(n_target × n_data)
    - Speed: 20-100x faster due to Numba JIT compilation
    - Scalability: Can handle millions of rows without OOM

    Args:
        target: Target DataFrame with interval columns
        data: Data DataFrame with values to aggregate
        join_left: Grouping columns (e.g., ['road', 'carriageway'])
        column_actions: List of Action objects specifying aggregations
        from_to: Tuple of (start_column, end_column) names
        verbose: Print progress information

    Returns:
        Target DataFrame with aggregated columns added (same structure as
        on_slk_intervals_optimized output)

    Example:
        >>> from pyroads.merge import on_slk_intervals_numba, Action, Aggregation
        >>> result = on_slk_intervals_numba(
        ...     target=target_df,
        ...     data=data_df,
        ...     join_left=["road", "cwy"],
        ...     column_actions=[
        ...         Action("roughness", Aggregation.LengthWeightedAverage(), "roughness_avg"),
        ...         Action("cracking", Aggregation.SumProportionOfTarget(), "cracking_pct"),
        ...     ],
        ...     from_to=("slk_from", "slk_to"),
        ... )
    """
    if not NUMBA_AVAILABLE:
        raise ImportError(
            "Numba is required for on_slk_intervals_numba. "
            "Install with: pip install numba>=0.57"
        )

    start_time = time.perf_counter()
    slk_from, slk_to = from_to

    # Validate inputs
    if not isinstance(join_left, list):
        raise TypeError("`join_left` must be a list of column names.")
    target = validation.ensure_dataframe("target", target)
    data = validation.ensure_dataframe("data", data)
    validation.ensure_simple_index("target", target)
    validation.ensure_simple_index("data", data)
    validation.ensure_required_columns(target, data, join_left, from_to)
    validation.ensure_nonzero_lengths("target", target, slk_from, slk_to)
    validation.ensure_nonzero_lengths("data", data, slk_from, slk_to)
    validation.ensure_output_columns_available(target.columns, column_actions)
    validation.ensure_aggregation_column_types(data, column_actions)

    # Initialize output columns with appropriate dtypes
    output_data: Dict[str, np.ndarray] = {}
    output_dtypes: Dict[str, str] = {}  # Track dtype for each output column
    for action in column_actions:
        # KeepLongest on non-numeric columns needs object dtype
        if action.aggregation.type.value == AGG_KEEP_LONGEST:
            if not _is_numeric_column(data[action.column_name]):
                output_data[action.rename] = np.full(len(target), None, dtype=object)
                output_dtypes[action.rename] = "object"
                continue
        # All other cases use float64
        output_data[action.rename] = np.full(len(target), np.nan, dtype=np.float64)
        output_dtypes[action.rename] = "float64"

    # Pre-extract needed columns from data
    data_needed_cols = list({action.column_name for action in column_actions})
    data_subset = data[[*join_left, slk_from, slk_to, *data_needed_cols]].copy()
    data_subset["_original_index"] = data.index
    data_subset["_segment_len"] = data_subset[slk_to] - data_subset[slk_from]

    # Build data groups
    data_groups: Dict[tuple, pd.DataFrame] = {}
    for key, group in data_subset.groupby(join_left, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        data_groups[key_tuple] = group.reset_index(drop=True)

    # Group target
    target_groups = target.groupby(join_left, sort=False)
    total_groups = target_groups.ngroups

    if verbose:
        print(
            f"[pyroads.merge] Numba sparse merge: {len(column_actions)} action(s), "
            f"{total_groups} group(s), {len(target)} target rows, {len(data)} data rows"
        )

    # Determine which columns need categorical handling
    categorical_actions = []
    numeric_actions = []
    for action in column_actions:
        if action.aggregation.type.value == AGG_KEEP_LONGEST:
            if not _is_numeric_column(data[action.column_name]):
                categorical_actions.append(action)
                continue
        numeric_actions.append(action)

    processed_groups = 0

    # Process each group
    for key, target_group in target_groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        data_group = data_groups.get(key_tuple)

        if data_group is None or len(data_group) == 0:
            processed_groups += 1
            continue

        # Extract numpy arrays
        tgt_starts = target_group[slk_from].to_numpy(dtype=np.float64)
        tgt_ends = target_group[slk_to].to_numpy(dtype=np.float64)
        tgt_lengths = tgt_ends - tgt_starts

        data_starts = data_group[slk_from].to_numpy(dtype=np.float64)
        data_ends = data_group[slk_to].to_numpy(dtype=np.float64)
        data_lengths = data_group["_segment_len"].to_numpy(dtype=np.float64)
        original_indices = data_group["_original_index"].to_numpy(dtype=np.int64)

        # Find sparse overlaps (THE KEY MEMORY OPTIMIZATION)
        tgt_idx, data_idx, overlap_lens = _find_overlapping_intervals_sorted(
            tgt_starts, tgt_ends, data_starts, data_ends
        )

        if len(tgt_idx) == 0:
            processed_groups += 1
            continue

        # Map local target indices to global target indices
        target_iloc_positions = [
            target.index.get_loc(idx) for idx in target_group.index
        ]

        # Process numeric columns with Numba
        for action in numeric_actions:
            col_values = data_group[action.column_name].to_numpy(dtype=np.float64)
            agg_type, percentile = _get_agg_type_code(action.aggregation)

            agg_results = _aggregate_all_targets_numeric(
                n_targets=len(target_group),
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

            # Map results to output array
            for local_idx, global_pos in enumerate(target_iloc_positions):
                output_data[action.rename][global_pos] = agg_results[local_idx]

        # Process categorical columns in Python
        for action in categorical_actions:
            col_values = data_group[action.column_name].to_numpy(dtype=object)

            agg_results = _aggregate_keep_longest_categorical(
                n_targets=len(target_group),
                tgt_indices=tgt_idx,
                data_indices=data_idx,
                overlap_lens=overlap_lens,
                col_values=col_values,
            )

            for local_idx, global_pos in enumerate(target_iloc_positions):
                output_data[action.rename][global_pos] = agg_results[local_idx]

        processed_groups += 1
        if verbose and processed_groups % 100 == 0:
            print(f"  Processed {processed_groups}/{total_groups} groups...")

    # Build result DataFrame
    result = target.copy()
    for col_name, col_values in output_data.items():
        # Convert back to appropriate dtype
        result[col_name] = pd.Series(col_values, index=target.index)

    elapsed = time.perf_counter() - start_time

    if verbose:
        print(f"[pyroads.merge] Numba sparse merge completed in {elapsed:.2f}s")

    # Emit performance metrics if logger is configured
    try:
        from . import merge as merge_module

        if hasattr(merge_module, "_emit_performance_event"):
            merge_module._emit_performance_event(
                "on_slk_intervals_numba",
                duration=elapsed,
                groups=float(total_groups),
                actions=float(len(column_actions)),
                rows=float(len(target)),
                overlaps=float(len(tgt_idx) if "tgt_idx" in dir() else 0),
            )
    except Exception:
        pass  # Performance logging is optional

    return result


def is_numba_available() -> bool:
    """Check if Numba is available for acceleration."""
    return NUMBA_AVAILABLE
