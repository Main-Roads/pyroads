"""
Tests for the Numba-accelerated sparse interval merge implementation.

These tests verify that on_slk_intervals_numba produces identical results
to on_slk_intervals_optimized while using significantly less memory.
"""

import numpy as np
import pandas as pd
import pytest

from pyroads.merge import (
    Action,
    Aggregation,
    is_numba_available,
    on_slk_intervals_numba,
    on_slk_intervals_optimized,
)


# Skip all tests if numba is not installed
pytestmark = pytest.mark.skipif(
    not is_numba_available(),
    reason="Numba not installed - install with: pip install numba>=0.57",
)


def make_test_data(
    n_groups: int = 3,
    targets_per_group: int = 100,
    data_per_group: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic test data."""
    rng = np.random.default_rng(seed)
    segment_len = 10

    target_records = []
    data_records = []

    for g in range(n_groups):
        road = f"R{g:03d}"
        # Target segments: consecutive non-overlapping
        start = 0
        for _ in range(targets_per_group):
            end = start + segment_len
            target_records.append((road, start, end))
            start = end

        total_length = targets_per_group * segment_len

        # Data segments: random overlapping
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


class TestNumbaVsOptimized:
    """Compare Numba implementation against optimized implementation."""

    def test_length_weighted_average(self):
        """LengthWeightedAverage should match between implementations."""
        target, data = make_test_data()

        actions = [
            Action("value", Aggregation.LengthWeightedAverage(), "value_lwa"),
        ]

        result_opt = on_slk_intervals_optimized(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )
        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        pd.testing.assert_series_equal(
            result_opt["value_lwa"].astype(float),
            result_numba["value_lwa"].astype(float),
            check_names=False,
            rtol=1e-10,
        )

    def test_sum(self):
        """Sum aggregation should match between implementations."""
        target, data = make_test_data()

        actions = [
            Action("value", Aggregation.Sum(), "value_sum"),
        ]

        result_opt = on_slk_intervals_optimized(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )
        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        pd.testing.assert_series_equal(
            result_opt["value_sum"].astype(float),
            result_numba["value_sum"].astype(float),
            check_names=False,
            rtol=1e-10,
        )

    def test_max_min(self):
        """Max and Min aggregations should match."""
        target, data = make_test_data()

        actions = [
            Action("value", Aggregation.Max(), "value_max"),
            Action("value", Aggregation.Min(), "value_min"),
        ]

        result_opt = on_slk_intervals_optimized(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )
        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        pd.testing.assert_series_equal(
            result_opt["value_max"].astype(float),
            result_numba["value_max"].astype(float),
            check_names=False,
            rtol=1e-10,
        )
        pd.testing.assert_series_equal(
            result_opt["value_min"].astype(float),
            result_numba["value_min"].astype(float),
            check_names=False,
            rtol=1e-10,
        )

    def test_average(self):
        """Simple Average should match."""
        target, data = make_test_data()

        actions = [
            Action("value", Aggregation.Average(), "value_avg"),
        ]

        result_opt = on_slk_intervals_optimized(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )
        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        pd.testing.assert_series_equal(
            result_opt["value_avg"].astype(float),
            result_numba["value_avg"].astype(float),
            check_names=False,
            rtol=1e-10,
        )

    def test_first(self):
        """First aggregation should match."""
        target, data = make_test_data()

        actions = [
            Action("value", Aggregation.First(), "value_first"),
        ]

        result_opt = on_slk_intervals_optimized(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )
        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        pd.testing.assert_series_equal(
            result_opt["value_first"].astype(float),
            result_numba["value_first"].astype(float),
            check_names=False,
            rtol=1e-10,
        )

    def test_sum_proportion_of_data(self):
        """SumProportionOfData should match."""
        target, data = make_test_data()

        actions = [
            Action("value", Aggregation.SumProportionOfData(), "value_spd"),
        ]

        result_opt = on_slk_intervals_optimized(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )
        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        pd.testing.assert_series_equal(
            result_opt["value_spd"].astype(float),
            result_numba["value_spd"].astype(float),
            check_names=False,
            rtol=1e-10,
        )

    def test_sum_proportion_of_target(self):
        """SumProportionOfTarget should match."""
        target, data = make_test_data()

        actions = [
            Action("load", Aggregation.SumProportionOfTarget(), "load_spt"),
        ]

        result_opt = on_slk_intervals_optimized(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )
        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        pd.testing.assert_series_equal(
            result_opt["load_spt"].astype(float),
            result_numba["load_spt"].astype(float),
            check_names=False,
            rtol=1e-10,
        )

    def test_length_weighted_percentile(self):
        """LengthWeightedPercentile should match."""
        target, data = make_test_data()

        actions = [
            Action(
                "value",
                Aggregation.LengthWeightedPercentile(0.5),
                "value_p50",
            ),
            Action(
                "value",
                Aggregation.LengthWeightedPercentile(0.9),
                "value_p90",
            ),
        ]

        result_opt = on_slk_intervals_optimized(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )
        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        pd.testing.assert_series_equal(
            result_opt["value_p50"].astype(float),
            result_numba["value_p50"].astype(float),
            check_names=False,
            rtol=1e-6,  # Slightly looser tolerance for percentile
        )
        pd.testing.assert_series_equal(
            result_opt["value_p90"].astype(float),
            result_numba["value_p90"].astype(float),
            check_names=False,
            rtol=1e-6,
        )

    def test_multiple_actions(self):
        """Multiple actions in single call should all match."""
        target, data = make_test_data()

        actions = [
            Action("value", Aggregation.LengthWeightedAverage(), "value_lwa"),
            Action("value", Aggregation.Sum(), "value_sum"),
            Action("load", Aggregation.Max(), "load_max"),
            Action("load", Aggregation.SumProportionOfTarget(), "load_spt"),
        ]

        result_opt = on_slk_intervals_optimized(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )
        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        for action in actions:
            pd.testing.assert_series_equal(
                result_opt[action.rename].astype(float),
                result_numba[action.rename].astype(float),
                check_names=False,
                rtol=1e-10,
            )

    def test_empty_groups(self):
        """Groups with no overlapping data should produce NaN."""
        target = pd.DataFrame(
            [
                ("R001", 0, 10),
                ("R001", 10, 20),
                ("R002", 0, 10),  # No data for R002
            ],
            columns=["road", "slk_from", "slk_to"],
        )

        data = pd.DataFrame(
            [
                ("R001", 5, 15, 100.0),
            ],
            columns=["road", "slk_from", "slk_to", "value"],
        )

        actions = [Action("value", Aggregation.Sum(), "value_sum")]

        result_numba = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        # R001 rows should have values, R002 should be NaN
        assert not pd.isna(result_numba.iloc[0]["value_sum"])
        assert not pd.isna(result_numba.iloc[1]["value_sum"])
        assert pd.isna(result_numba.iloc[2]["value_sum"])


class TestNumbaSparseOverlap:
    """Test the sparse overlap computation specifically."""

    def test_no_overlaps(self):
        """Non-overlapping intervals should produce empty results."""
        target = pd.DataFrame(
            [("R001", 0, 10), ("R001", 20, 30)],
            columns=["road", "slk_from", "slk_to"],
        )
        data = pd.DataFrame(
            [("R001", 10, 20, 50.0)],  # Between target segments
            columns=["road", "slk_from", "slk_to", "value"],
        )

        actions = [Action("value", Aggregation.Sum(), "value_sum")]
        result = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        assert pd.isna(result.iloc[0]["value_sum"])
        assert pd.isna(result.iloc[1]["value_sum"])

    def test_partial_overlap(self):
        """Partial overlaps should be correctly computed."""
        target = pd.DataFrame(
            [("R001", 0, 100)],
            columns=["road", "slk_from", "slk_to"],
        )
        data = pd.DataFrame(
            [("R001", 50, 150, 100.0)],  # 50% overlap
            columns=["road", "slk_from", "slk_to", "value"],
        )

        actions = [Action("value", Aggregation.SumProportionOfData(), "value_spd")]
        result = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        # Overlap is 50 out of data length 100, so proportion = 0.5
        # Result should be 100 * 0.5 = 50
        assert np.isclose(result.iloc[0]["value_spd"], 50.0)


class TestNumbaPerformance:
    """Basic performance sanity checks."""

    @pytest.mark.slow
    def test_large_dataset_no_oom(self):
        """Verify large dataset doesn't cause OOM (the main goal)."""
        # This would cause ~800MB matrix with dense approach
        # With sparse approach, should use ~10-50MB
        target, data = make_test_data(
            n_groups=10,
            targets_per_group=1000,
            data_per_group=5000,
        )

        actions = [
            Action("value", Aggregation.LengthWeightedAverage(), "value_lwa"),
            Action("load", Aggregation.SumProportionOfTarget(), "load_spt"),
        ]

        # This should complete without OOM
        result = on_slk_intervals_numba(
            target, data, ["road"], actions, ("slk_from", "slk_to")
        )

        assert len(result) == len(target)
        assert "value_lwa" in result.columns
        assert "load_spt" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
