"""Repeatable segmenter timing harness.

Run from the pyroads project with ``uv run python examples/segmenter/benchmark.py``.
The benchmark warms Numba before collecting timings and prints JSON so baseline and
optimized runs can be compared without changing package code.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from pyroads.segmenter import (
    cross_sections,
    segment_by_categories_and_slk_discontinuities,
    segment_by_categories_and_slk_true_discontinuities,
    split_rows_by_category_to_max_segment_length,
    split_rows_by_segmentation,
)


CATEGORIES = ["road", "lane"]
MEASURE_SLK = ("slk_from", "slk_to")
MEASURE_TRUE = ("true_from", "true_to")


def _category_data(rows: int, groups: int = 8) -> pd.DataFrame:
    rows_per_group = rows // groups
    parts = []
    for group in range(groups):
        starts = np.arange(rows_per_group, dtype=float) * 0.01
        parts.append(pd.DataFrame({
            "road": f"R{group}",
            "lane": "L",
            "slk_from": starts,
            "slk_to": starts + 0.01,
            "true_from": starts,
            "true_to": starts + 0.01,
        }))
    return pd.concat(parts, ignore_index=True)


def _overlay_data(rows: int, groups: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = _category_data(rows, groups)
    additional = original.iloc[::2].copy().reset_index(drop=True)
    return original, additional


def _cross_section_data(rows: int, lanes: int = 4) -> pd.DataFrame:
    starts = np.arange(rows, dtype=float) * 0.01
    return pd.concat([
        pd.DataFrame({
            "road": "R0",
            "lane": f"L{lane}",
            "slk_from": starts,
            "slk_to": starts + 0.01,
            "true_from": starts,
            "true_to": starts + 0.01,
        })
        for lane in range(lanes)
    ], ignore_index=True)


def _time_call(function, repeats: int) -> tuple[float, int]:
    result = function()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        timings.append((time.perf_counter() - started) * 1000)
    return float(np.median(timings)), len(result)


def run(repeats: int) -> dict[str, dict[str, float | int]]:
    category = _category_data(128_000)
    original, additional = _overlay_data(8_000)
    cross_section = _cross_section_data(1_000)
    max_split = _category_data(32_000)

    measurements = {
        "category_slk": lambda: segment_by_categories_and_slk_discontinuities(
            category, CATEGORIES, MEASURE_SLK
        ),
        "category_true": lambda: segment_by_categories_and_slk_true_discontinuities(
            category, CATEGORIES, MEASURE_SLK, MEASURE_TRUE
        ),
        "overlay": lambda: split_rows_by_segmentation(
            original, additional, CATEGORIES, MEASURE_SLK, MEASURE_TRUE,
            "original_index", "additional_index",
        ),
        "cross_sections": lambda: cross_sections(
            cross_section, ["road"], ["lane"], MEASURE_SLK, MEASURE_TRUE
        ),
        "max_length": lambda: split_rows_by_category_to_max_segment_length(
            max_split, MEASURE_SLK, MEASURE_TRUE, CATEGORIES, 0.1
        ),
    }
    return {
        name: {
            "median_ms": median_ms,
            "output_rows": output_rows,
        }
        for name, function in measurements.items()
        for median_ms, output_rows in [_time_call(function, repeats)]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run(args.repeats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
