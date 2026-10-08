"""Compare interval merge timings for pandas and Polars inputs.

Usage:
    python examples/merge/benchmark_backends.py [--roads 2000] [--repeats 3]

Each road has two carriageways, giving ``2 * roads`` join groups with 100
target rows and 400 data rows per carriageway.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import polars as pl

from pyroads.merge import Action, Aggregation, on_slk_intervals

NUMERIC = [
    Action("rough", Aggregation.LengthWeightedAverage(), "rough_lwa"),
    Action("rut", Aggregation.Max(), "rut_max"),
    Action("rough", Aggregation.LengthWeightedPercentile(0.9), "rough_p90"),
]
CATEGORICAL = [Action("surf", Aggregation.KeepLongest(), "surf_kl")]


def make_frames(roads: int, targets_per_group: int = 100, data_per_group: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)

    def build(rows_per_group: int, step: float):
        road = np.repeat(np.arange(roads), 2 * rows_per_group)
        cwy = np.tile(np.repeat(["L", "R"], rows_per_group), roads)
        start = np.tile(np.arange(rows_per_group) * step, 2 * roads)
        return road, cwy, start

    road, cwy, start = build(targets_per_group, 100.0)
    target = pd.DataFrame({"road": road, "cwy": cwy, "slk_from": start, "slk_to": start + 100.0})
    step = 100.0 * targets_per_group / data_per_group
    road, cwy, start = build(data_per_group, step)
    data = pd.DataFrame(
        {
            "road": road,
            "cwy": cwy,
            "slk_from": start,
            "slk_to": start + step,
            "rough": rng.normal(3, 1, len(road)),
            "rut": rng.normal(5, 2, len(road)),
            "surf": rng.choice(["AC", "SS", "PM"], len(road)),
        }
    )
    return target, data


def to_polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        {
            column: frame[column].to_numpy() if frame[column].dtype.kind in "if" else frame[column].astype(str).tolist()
            for column in frame.columns
        }
    )


def best_time(function, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def compare(roads: int, repeats: int = 3) -> list[tuple[str, float, float]]:
    target, data = make_frames(roads)
    target_pl, data_pl = to_polars(target), to_polars(data)
    rows = []
    for label, actions in (("numeric", NUMERIC), ("numeric+categorical", NUMERIC + CATEGORICAL)):
        arguments = dict(join_left=["road", "cwy"], column_actions=actions, from_to=("slk_from", "slk_to"))
        on_slk_intervals(target.head(10), data.head(40), **arguments)
        pandas_time = best_time(lambda: on_slk_intervals(target, data, **arguments), repeats)
        polars_time = best_time(lambda: on_slk_intervals(target_pl, data_pl, **arguments), repeats)
        rows.append((label, pandas_time, polars_time))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roads", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()

    target, data = make_frames(arguments.roads)
    print(f"{2 * arguments.roads} groups, {len(target)} target rows, {len(data)} data rows")
    print(f"{'actions':22s} {'pandas':>9s} {'polars':>9s} {'ratio':>7s}")
    for label, pandas_time, polars_time in compare(arguments.roads, arguments.repeats):
        print(f"{label:22s} {pandas_time:8.3f}s {polars_time:8.3f}s {polars_time / pandas_time:6.2f}x")


if __name__ == "__main__":
    main()
