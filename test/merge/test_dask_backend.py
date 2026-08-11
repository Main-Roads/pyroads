"""Tests for the Dask-native, out-of-core interval merge implementation.

These tests verify that on_slk_intervals_dask produces results identical to
the pandas/Numba path after `.compute()`, that the result stays lazy until
computed, and that the type-detection dispatcher correctly routes Dask
inputs to it.
"""

import numpy as np
import pandas as pd
import pytest

from pyroads.merge import (
    Action,
    Aggregation,
    is_dask_available,
    on_slk_intervals_auto,
    on_slk_intervals_numba,
)

dd = pytest.importorskip("dask.dataframe")

pytestmark = pytest.mark.skipif(
    not is_dask_available(),
    reason="dask not installed - install with: pip install pyroads.merge[dask]",
)

from pyroads.merge._dask_merge import on_slk_intervals_dask  # noqa: E402


def make_test_data(
    n_groups: int = 4,
    targets_per_group: int = 40,
    data_per_group: int = 150,
    seed: int = 7,
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

        for s, e, v in zip(starts, ends, values):
            data_records.append((road, int(s), int(e), float(v)))

    target = pd.DataFrame(target_records, columns=["road", "slk_from", "slk_to"])
    data = pd.DataFrame(data_records, columns=["road", "slk_from", "slk_to", "value"])
    return target, data


class TestDaskVsNumba:
    def test_result_stays_lazy_until_compute(self):
        target, data = make_test_data()
        actions = [Action("value", Aggregation.LengthWeightedAverage(), "value_lwa")]

        target_dd = dd.from_pandas(target, npartitions=4)
        data_dd = dd.from_pandas(data, npartitions=4)

        lazy_result = on_slk_intervals_dask(
            target_dd, data_dd, ["road"], actions, ("slk_from", "slk_to")
        )
        assert isinstance(lazy_result, dd.DataFrame)
        # Should not have been computed yet -- computing explicitly here.
        computed = lazy_result.compute()
        assert isinstance(computed, pd.DataFrame)
        assert len(computed) == len(target)

    @pytest.mark.parametrize(
        "aggregation,rename",
        [
            (Aggregation.LengthWeightedAverage(), "value_lwa"),
            (Aggregation.Sum(), "value_sum"),
            (Aggregation.Min(), "value_min"),
            (Aggregation.Max(), "value_max"),
            (Aggregation.Average(), "value_avg"),
        ],
    )
    def test_numeric_aggregations_match(self, aggregation, rename):
        target, data = make_test_data()
        actions = [Action("value", aggregation, rename)]

        result_numba = (
            on_slk_intervals_numba(
                target, data, ["road"], actions, ("slk_from", "slk_to")
            )
            .sort_values(["road", "slk_from"])
            .reset_index(drop=True)
        )

        target_dd = dd.from_pandas(target, npartitions=3)
        data_dd = dd.from_pandas(data, npartitions=5)
        result_dask = (
            on_slk_intervals_dask(
                target_dd, data_dd, ["road"], actions, ("slk_from", "slk_to")
            )
            .compute()
            .sort_values(["road", "slk_from"])
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(
            result_dask[rename].to_numpy(),
            result_numba[rename].to_numpy(),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )

    def test_single_partition_matches_multi_partition(self):
        target, data = make_test_data(
            n_groups=3, targets_per_group=30, data_per_group=100
        )
        actions = [Action("value", Aggregation.LengthWeightedAverage(), "value_lwa")]

        target_dd = dd.from_pandas(target, npartitions=1)
        data_dd = dd.from_pandas(data, npartitions=1)
        result_single = (
            on_slk_intervals_dask(
                target_dd, data_dd, ["road"], actions, ("slk_from", "slk_to")
            )
            .compute()
            .sort_values(["road", "slk_from"])
            .reset_index(drop=True)
        )

        target_dd_multi = dd.from_pandas(target, npartitions=4)
        data_dd_multi = dd.from_pandas(data, npartitions=6)
        result_multi = (
            on_slk_intervals_dask(
                target_dd_multi,
                data_dd_multi,
                ["road"],
                actions,
                ("slk_from", "slk_to"),
            )
            .compute()
            .sort_values(["road", "slk_from"])
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(
            result_single["value_lwa"].to_numpy(),
            result_multi["value_lwa"].to_numpy(),
            equal_nan=True,
        )


class TestAutoDispatchRoutesToDask:
    def test_on_slk_intervals_auto_detects_dask(self):
        target, data = make_test_data(
            n_groups=2, targets_per_group=15, data_per_group=60
        )
        actions = [Action("value", Aggregation.Sum(), "value_sum")]

        result_pandas = (
            on_slk_intervals_numba(
                target, data, ["road"], actions, ("slk_from", "slk_to")
            )
            .sort_values(["road", "slk_from"])
            .reset_index(drop=True)
        )

        target_dd = dd.from_pandas(target, npartitions=2)
        data_dd = dd.from_pandas(data, npartitions=2)
        result_auto = on_slk_intervals_auto(
            target_dd, data_dd, ["road"], actions, ("slk_from", "slk_to")
        )
        assert isinstance(result_auto, dd.DataFrame)
        computed = (
            result_auto.compute()
            .sort_values(["road", "slk_from"])
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(
            computed["value_sum"].to_numpy(),
            result_pandas["value_sum"].to_numpy(),
            equal_nan=True,
        )

    def test_mixed_backend_types_raise_type_error(self):
        target, data = make_test_data(
            n_groups=1, targets_per_group=5, data_per_group=10
        )
        actions = [Action("value", Aggregation.Sum(), "value_sum")]

        target_dd = dd.from_pandas(target, npartitions=1)
        with pytest.raises(TypeError):
            on_slk_intervals_auto(
                target_dd,
                data,  # pandas, mismatched with dask target
                ["road"],
                actions,
                ("slk_from", "slk_to"),
            )
