"""Tests for the Polars-native interval merge implementation.

These tests verify that on_slk_intervals_polars produces results identical
to the pandas/Numba path (on_slk_intervals_numba), and that the
type-detection dispatcher in on_slk_intervals_auto correctly routes Polars
inputs to it.
"""

import numpy as np
import pandas as pd
import pytest

from pyroads.merge import (
    Action,
    Aggregation,
    is_polars_available,
    on_slk_intervals_auto,
    on_slk_intervals_numba,
)

pl = pytest.importorskip("polars")

pytestmark = pytest.mark.skipif(
    not is_polars_available(),
    reason="polars not installed - install with: pip install pyroads.merge[polars]",
)

from pyroads.merge._polars_merge import on_slk_intervals_polars  # noqa: E402


def make_test_data(
    n_groups: int = 3,
    targets_per_group: int = 100,
    data_per_group: int = 500,
    seed: int = 42,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Generate synthetic test data (mirrors tests/test_numba_merge.py)."""
    rng = np.random.default_rng(seed)
    segment_len = 10

    target_records = []
    data_records = []

    for g in range(n_groups):
        road = f"R{g:03d}"
        start = 0
        for _ in range(targets_per_group):
            end = start + segment_len
            target_records.append((road, start, end))
            start = end

        total_length = targets_per_group * segment_len

        starts = rng.integers(0, total_length - 5, size=data_per_group)
        lengths = rng.integers(5, segment_len * 3, size=data_per_group)
        ends = np.minimum(starts + lengths, total_length)
        ends = np.where(ends == starts, ends + 1, ends)

        values = rng.random(data_per_group) * 100
        loads = rng.random(data_per_group) * 10
        categories = rng.choice(["A", "B", "C", "D"], size=data_per_group)

        for s, e, v, ld, cat in zip(starts, ends, values, loads, categories):
            data_records.append((road, int(s), int(e), float(v), float(ld), cat))

    target = pd.DataFrame(target_records, columns=["road", "slk_from", "slk_to"])
    data = pd.DataFrame(
        data_records,
        columns=["road", "slk_from", "slk_to", "value", "load", "category"],
    )
    return target, data


class TestPolarsVsNumba:
    """Compare the Polars-native implementation against the pandas/Numba path."""

    @pytest.mark.parametrize(
        "aggregation,rename",
        [
            (Aggregation.LengthWeightedAverage(), "value_lwa"),
            (Aggregation.Average(), "value_avg"),
            (Aggregation.Sum(), "value_sum"),
            (Aggregation.Min(), "value_min"),
            (Aggregation.Max(), "value_max"),
            (Aggregation.SumProportionOfData(), "value_spd"),
            (Aggregation.SumProportionOfTarget(), "value_spt"),
            (Aggregation.First(), "value_first"),
            (Aggregation.KeepLongest(), "value_keep_longest"),
            (Aggregation.LengthWeightedPercentile(0.5), "value_p50"),
        ],
    )
    def test_numeric_aggregations_match(self, aggregation, rename):
        target, data = make_test_data()
        actions = [Action("value", aggregation, rename)]

        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        target_pl = pl.from_pandas(target)
        data_pl = pl.from_pandas(data)
        result_polars = on_slk_intervals_polars(
            target_pl, data_pl, ["road"], actions, ("slk_from", "slk_to")
        )

        np.testing.assert_allclose(
            result_polars[rename].to_numpy(),
            result_numba[rename].to_numpy(),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )

    def test_categorical_keep_longest_matches(self):
        target, data = make_test_data()
        actions = [Action("category", Aggregation.KeepLongest(), "category_kl")]

        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )
        target_pl = pl.from_pandas(target)
        data_pl = pl.from_pandas(data)
        result_polars = on_slk_intervals_polars(
            target_pl, data_pl, ["road"], actions, ("slk_from", "slk_to")
        )

        expected = result_numba["category_kl"].tolist()
        actual = result_polars["category_kl"].to_list()
        assert actual == expected

    def test_multiple_actions_and_groups_match(self):
        target, data = make_test_data(
            n_groups=5, targets_per_group=50, data_per_group=200
        )
        actions = [
            Action("value", Aggregation.LengthWeightedAverage(), "value_lwa"),
            Action("load", Aggregation.Sum(), "load_sum"),
            Action("category", Aggregation.KeepLongest(), "category_kl"),
        ]

        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )
        target_pl = pl.from_pandas(target)
        data_pl = pl.from_pandas(data)
        result_polars = on_slk_intervals_polars(
            target_pl, data_pl, ["road"], actions, ("slk_from", "slk_to")
        )

        np.testing.assert_allclose(
            result_polars["value_lwa"].to_numpy(),
            result_numba["value_lwa"].to_numpy(),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            result_polars["load_sum"].to_numpy(),
            result_numba["load_sum"].to_numpy(),
            equal_nan=True,
        )
        assert (
            result_polars["category_kl"].to_list()
            == result_numba["category_kl"].tolist()
        )

    def test_single_thread_matches_multi_thread(self):
        """n_jobs=1 (sequential) must match the default threaded execution."""
        target, data = make_test_data(
            n_groups=6, targets_per_group=40, data_per_group=150
        )
        actions = [Action("value", Aggregation.LengthWeightedAverage(), "value_lwa")]

        target_pl = pl.from_pandas(target)
        data_pl = pl.from_pandas(data)

        result_seq = on_slk_intervals_polars(
            target_pl, data_pl, ["road"], actions, ("slk_from", "slk_to"), n_jobs=1
        )
        result_par = on_slk_intervals_polars(
            target_pl, data_pl, ["road"], actions, ("slk_from", "slk_to"), n_jobs=4
        )

        np.testing.assert_allclose(
            result_seq["value_lwa"].to_numpy(),
            result_par["value_lwa"].to_numpy(),
            equal_nan=True,
        )

    def test_no_overlapping_data_group_is_nan(self):
        target = pd.DataFrame({"road": ["R1"], "slk_from": [0], "slk_to": [10]})
        data = pd.DataFrame(
            {"road": ["R2"], "slk_from": [0], "slk_to": [10], "value": [5.0]}
        )
        actions = [Action("value", Aggregation.Sum(), "value_sum")]

        result = on_slk_intervals_polars(
            pl.from_pandas(target),
            pl.from_pandas(data),
            ["road"],
            actions,
            ("slk_from", "slk_to"),
        )
        assert np.isnan(result["value_sum"].to_numpy()).all()


class TestAutoDispatchRoutesToPolars:
    def test_on_slk_intervals_auto_detects_polars(self):
        target, data = make_test_data(
            n_groups=2, targets_per_group=20, data_per_group=80
        )
        actions = [Action("value", Aggregation.Sum(), "value_sum")]

        result_pandas = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )
        result_auto = on_slk_intervals_auto(
            pl.from_pandas(target),
            pl.from_pandas(data),
            ["road"],
            actions,
            ("slk_from", "slk_to"),
        )
        assert isinstance(result_auto, pl.DataFrame)
        np.testing.assert_allclose(
            result_auto["value_sum"].to_numpy(),
            result_pandas["value_sum"].to_numpy(),
            equal_nan=True,
        )

    def test_mixed_backend_types_raise_type_error(self):
        target, data = make_test_data(
            n_groups=1, targets_per_group=5, data_per_group=10
        )
        actions = [Action("value", Aggregation.Sum(), "value_sum")]

        with pytest.raises(TypeError):
            on_slk_intervals_auto(
                pl.from_pandas(target),
                data,  # pandas, mismatched with polars target
                ["road"],
                actions,
                ("slk_from", "slk_to"),
            )
