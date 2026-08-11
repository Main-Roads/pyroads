"""Benchmark pyroads.merge legacy vs optimized implementations.

Usage:
    # Use synthetic data (default)
    python benchmarks/compare_merges.py --targets 5000 --data 15000 --groups 5 --repeats 5

    # Use real data from CSV files
    python benchmarks/compare_merges.py --target-file target.csv --data-file data.csv --repeats 5

The script can use either synthetic interval data or load real data from CSV files,
executes both merge paths, asserts that their outputs match, and prints timing statistics.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pyroads.merge import merge as merge_module


@dataclass
class BenchmarkConfig:
    target_rows: int
    data_rows: int
    groups: int
    repeats: int
    seed: int
    target_file: str | None = None
    data_file: str | None = None


def _build_frames(
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[merge_module.Action]]:
    rng = np.random.default_rng(config.seed)
    segment_length = 100
    target_rows_per_group = config.target_rows // config.groups
    data_rows_per_group = config.data_rows // config.groups

    target_records = []
    data_records = []

    for group_idx in range(config.groups):
        road = f"R{group_idx:03d}"
        start = 0
        for _ in range(target_rows_per_group):
            end = start + segment_length
            target_records.append((road, start, end))
            start = end

        total_length = target_rows_per_group * segment_length
        starts = rng.integers(0, total_length - 5, size=data_rows_per_group)
        lengths = rng.integers(5, segment_length * 2, size=data_rows_per_group)
        ends = np.minimum(starts + lengths, total_length)
        # ensure non-zero length
        ends = np.where(ends == starts, ends + 1, ends)

        values = rng.random(size=data_rows_per_group) * 100
        load = rng.random(size=data_rows_per_group) * 10

        for s, e, val, load_val in zip(starts, ends, values, load):
            data_records.append((road, int(s), int(e), float(val), float(load_val)))

    target = pd.DataFrame(target_records, columns=["road", "slk_from", "slk_to"])
    data = pd.DataFrame(
        data_records,
        columns=["road", "slk_from", "slk_to", "value", "load"],
    )

    actions = [
        merge_module.Action(
            "value",
            rename="value_avg",
            aggregation=merge_module.Aggregation.LengthWeightedAverage(),
        ),
        merge_module.Action(
            "value",
            rename="value_pct90",
            aggregation=merge_module.Aggregation.LengthWeightedPercentile(0.90),
        ),
        merge_module.Action(
            "load",
            rename="load_target_sum",
            aggregation=merge_module.Aggregation.SumProportionOfTarget(),
        ),
    ]

    return target, data, actions


def _time_function(func, *args, repeats: int) -> tuple[float, pd.DataFrame]:
    best_elapsed = float("inf")
    best_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - start
        if elapsed < best_elapsed:
            best_elapsed = elapsed
            best_result = result
    assert best_result is not None
    return best_elapsed, best_result


def _normalize_column_name(col: str) -> str:
    """Normalize column name to lowercase and remove common separators."""
    return col.lower().replace("_", "").replace(" ", "")


def _detect_road_column(columns: list[str]) -> str | None:
    """Detect road number/name column with various naming patterns."""
    road_patterns = ["roadnumber", "roadname", "roadno", "roadnum", "road"]
    normalized_cols = {_normalize_column_name(c): c for c in columns}

    for pattern in road_patterns:
        if pattern in normalized_cols:
            return normalized_cols[pattern]
    return None


def _detect_carriageway_column(columns: list[str]) -> str | None:
    """Detect carriageway column with various naming patterns."""
    cwy_patterns = ["carriageway", "cway", "cwy"]
    normalized_cols = {_normalize_column_name(c): c for c in columns}

    for pattern in cwy_patterns:
        if pattern in normalized_cols:
            return normalized_cols[pattern]
    return None


def _detect_xsp_column(columns: list[str]) -> str | None:
    """Detect XSP column."""
    xsp_patterns = ["xsp"]
    normalized_cols = {_normalize_column_name(c): c for c in columns}

    for pattern in xsp_patterns:
        if pattern in normalized_cols:
            return normalized_cols[pattern]
    return None


def _detect_from_to_columns(columns: list[str]) -> tuple[str, str] | None:
    """Detect from/to columns with various naming patterns."""
    normalized_cols = {_normalize_column_name(c): c for c in columns}

    # Patterns for 'from' column
    from_patterns = [
        "slkfrom",
        "fromslk",
        "startslk",
        "slkstart",
        "truefrom",
        "fromtrue",
        "starttrue",
        "truestart",
        "from",
        "start",
    ]

    # Patterns for 'to' column
    to_patterns = [
        "slkto",
        "toslk",
        "endslk",
        "slkend",
        "trueto",
        "totrue",
        "endtrue",
        "trueend",
        "to",
        "end",
    ]

    from_col = None
    to_col = None

    # Find 'from' column
    for pattern in from_patterns:
        if pattern in normalized_cols:
            from_col = normalized_cols[pattern]
            break

    # Find 'to' column
    for pattern in to_patterns:
        if pattern in normalized_cols:
            to_col = normalized_cols[pattern]
            break

    if from_col and to_col:
        return (from_col, to_col)
    return None


def run_benchmark(config: BenchmarkConfig) -> None:
    # Load or generate data
    if config.target_file and config.data_file:
        print("Loading data from files:")
        print(f"  Target: {config.target_file}")
        print(f"  Data  : {config.data_file}")
        target = pd.read_csv(config.target_file)
        data = pd.read_csv(config.data_file)

        # Detect from/to columns in target
        from_to = _detect_from_to_columns(target.columns.tolist())
        if not from_to:
            raise ValueError(
                "Could not identify 'from' and 'to' columns in target data. "
                "Expected patterns like: slk_from/slk_to, start_slk/end_slk, "
                "true_from/true_to, etc."
            )

        # Also detect from/to columns in data to exclude them from aggregation
        data_from_to = _detect_from_to_columns(data.columns.tolist())

        # Detect standard join columns (road, carriageway, xsp)
        join_left = []

        # Check both target and data for road column
        road_col_target = _detect_road_column(target.columns.tolist())
        road_col_data = _detect_road_column(data.columns.tolist())
        if road_col_target and road_col_data:
            # If different names, rename data column to match target
            if road_col_target != road_col_data:
                data = data.rename(columns={road_col_data: road_col_target})
            join_left.append(road_col_target)

        # Check both target and data for carriageway column
        cwy_col_target = _detect_carriageway_column(target.columns.tolist())
        cwy_col_data = _detect_carriageway_column(data.columns.tolist())
        if cwy_col_target and cwy_col_data:
            # If different names, rename data column to match target
            if cwy_col_target != cwy_col_data:
                data = data.rename(columns={cwy_col_data: cwy_col_target})
            join_left.append(cwy_col_target)

        # Check both target and data for xsp column
        xsp_col_target = _detect_xsp_column(target.columns.tolist())
        xsp_col_data = _detect_xsp_column(data.columns.tolist())
        if xsp_col_target and xsp_col_data:
            # If different names, rename data column to match target
            if xsp_col_target != xsp_col_data:
                data = data.rename(columns={xsp_col_data: xsp_col_target})
            join_left.append(xsp_col_target)

        if not join_left:
            raise ValueError(
                "Could not identify common join columns. "
                "Expected at least one of: road_no/road_name, carriageway/cway, xsp"
            )

        # Actions: aggregate all numeric columns from data (excluding join and from/to)
        excluded_cols = set(join_left) | set(from_to)
        if data_from_to:
            excluded_cols |= set(data_from_to)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        action_cols = [c for c in numeric_cols if c not in excluded_cols]
        if not action_cols:
            raise ValueError("No numeric columns found in data for aggregation")

        actions = [
            merge_module.Action(
                col,
                aggregation=merge_module.Aggregation.LengthWeightedAverage(),
            )
            for col in action_cols
        ]

        print(f"  Join columns: {join_left}")
        print(f"  Interval columns: {from_to}")
        print(f"  Aggregating: {action_cols}")
        print()
    else:
        target, data, actions = _build_frames(config)
        join_left = ["road"]
        from_to = ("slk_from", "slk_to")

    legacy_time, legacy_result = _time_function(
        merge_module.on_slk_intervals,
        target,
        data,
        join_left,
        actions,
        from_to,
        repeats=config.repeats,
    )

    optimized_time, optimized_result = _time_function(
        merge_module.on_slk_intervals_optimized,
        target,
        data,
        join_left,
        actions,
        from_to,
        repeats=config.repeats,
    )

    assert_frame_equal(
        legacy_result.sort_index(axis=1), optimized_result.sort_index(axis=1)
    )

    speedup = legacy_time / optimized_time if optimized_time else float("inf")

    print("Benchmark results (best of repeats):")
    print(f"  Legacy   : {legacy_time:.2f} s")
    print(f"  Optimized: {optimized_time:.2f} s")
    print(f"  Speedup  : {speedup:.2f}x")

    # Numba benchmark (optional)
    if getattr(merge_module, "is_numba_available", lambda: False)():
        # Warm-up to trigger JIT compilation outside timing loop
        try:
            merge_module.on_slk_intervals_numba(
                target, data, join_left, actions, from_to, verbose=False
            )
        except Exception:
            # Ignore warm-up errors - timed run below will surface any problems
            pass

        numba_time, numba_result = _time_function(
            merge_module.on_slk_intervals_numba,
            target,
            data,
            join_left,
            actions,
            from_to,
            repeats=config.repeats,
        )

        # Validate results match existing implementations
        assert_frame_equal(
            legacy_result.sort_index(axis=1), numba_result.sort_index(axis=1)
        )

        numba_speedup_vs_legacy = (
            legacy_time / numba_time if numba_time else float("inf")
        )
        numba_speedup_vs_optimized = (
            optimized_time / numba_time if numba_time else float("inf")
        )

        print(f"  Numba    : {numba_time:.2f} s")
        print(f"  Speedup vs Legacy   : {numba_speedup_vs_legacy:.2f}x")
        print(f"  Speedup vs Optimized: {numba_speedup_vs_optimized:.2f}x")
    else:
        print("  Numba    : (unavailable) - skipping Numba benchmark")


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark pyroads.merge helpers")
    parser.add_argument(
        "--targets", type=int, default=5000, help="Total target rows (synthetic mode)"
    )
    parser.add_argument(
        "--data", type=int, default=15000, help="Total data rows (synthetic mode)"
    )
    parser.add_argument(
        "--groups", type=int, default=5, help="Number of join groups (synthetic mode)"
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="Repeat runs and keep best timing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (synthetic mode)",
    )
    parser.add_argument(
        "--target-file", type=str, help="Path to target CSV file (real data mode)"
    )
    parser.add_argument(
        "--data-file", type=str, help="Path to data CSV file (real data mode)"
    )
    args = parser.parse_args()

    # Validate: either both files provided or neither
    if bool(args.target_file) != bool(args.data_file):
        parser.error("Both --target-file and --data-file must be provided together")

    return BenchmarkConfig(
        target_rows=args.targets,
        data_rows=args.data,
        groups=args.groups,
        repeats=args.repeats,
        seed=args.seed,
        target_file=args.target_file,
        data_file=args.data_file,
    )


if __name__ == "__main__":
    run_benchmark(parse_args())
