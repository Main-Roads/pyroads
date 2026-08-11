import pandas as pd
import pytest

from pyroads.merge import merge as merge


@pytest.fixture
def percentile_frames():
    target = pd.DataFrame(
        columns=["road", "slk_from", "slk_to"],
        data=[["H001", 0, 10]],
    )
    data = pd.DataFrame(
        columns=["road", "slk_from", "slk_to", "value"],
        data=[
            ["H001", 0, 4, 10.0],
            ["H001", 4, 10, 20.0],
        ],
    )
    return target, data


@pytest.mark.parametrize(
    ("percentile", "expected"),
    [(0.0, 10.0), (1.0, 20.0)],
)
def test_percentile_min_max_legacy(percentile, expected, percentile_frames):
    target, data = percentile_frames
    result = merge.on_slk_intervals(
        target=target,
        data=data,
        join_left=["road"],
        column_actions=[
            merge.Action(
                "value",
                rename=f"pct_{int(percentile * 100)}",
                aggregation=merge.Aggregation.LengthWeightedPercentile(percentile),
            )
        ],
        from_to=("slk_from", "slk_to"),
    )
    assert result.iloc[0, -1] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("percentile", "expected"),
    [(0.0, 10.0), (1.0, 20.0)],
)
def test_percentile_min_max_optimized(percentile, expected, percentile_frames):
    target, data = percentile_frames
    result = merge.on_slk_intervals_optimized(
        target=target,
        data=data,
        join_left=["road"],
        column_actions=[
            merge.Action(
                "value",
                rename=f"pct_{int(percentile * 100)}",
                aggregation=merge.Aggregation.LengthWeightedPercentile(percentile),
            )
        ],
        from_to=("slk_from", "slk_to"),
    )
    assert result.iloc[0, -1] == pytest.approx(expected)


def test_percentile_empty_overlap_returns_nan():
    target = pd.DataFrame(
        columns=["road", "slk_from", "slk_to"],
        data=[["H001", 100, 110]],
    )
    data = pd.DataFrame(
        columns=["road", "slk_from", "slk_to", "value"],
        data=[["H001", 0, 10, 5.0]],
    )
    action = merge.Action(
        "value",
        rename="pct_50",
        aggregation=merge.Aggregation.LengthWeightedPercentile(0.5),
    )
    legacy = merge.on_slk_intervals(
        target=target,
        data=data,
        join_left=["road"],
        column_actions=[action],
        from_to=("slk_from", "slk_to"),
    )
    optimized = merge.on_slk_intervals_optimized(
        target=target,
        data=data,
        join_left=["road"],
        column_actions=[action],
        from_to=("slk_from", "slk_to"),
    )
    assert pd.isna(legacy.loc[0, "pct_50"])
    assert pd.isna(optimized.loc[0, "pct_50"])
