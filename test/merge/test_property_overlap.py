import random

import pandas as pd
import pytest

from pyroads.merge import merge as merge_module


def _generate_segments(rng: random.Random, group_count: int, per_group: int):
    records = []
    for group_idx in range(group_count):
        road = f"R{group_idx:03d}"
        start = 0
        for _ in range(per_group):
            length = rng.randint(5, 50)
            end = start + length
            records.append((road, start, end))
            start = end
    return pd.DataFrame(records, columns=["road", "slk_from", "slk_to"])


def _generate_data_segments(
    rng: random.Random, group_count: int, count: int, max_len: int
):
    records = []
    for group_idx in range(group_count):
        road = f"R{group_idx:03d}"
        for _ in range(count):
            start = rng.randint(0, max_len - 1)
            length = rng.randint(1, max_len)
            end = min(start + length, max_len)
            if start == end:
                end += 1
            value = rng.random() * 100
            records.append((road, start, end, value))
    return pd.DataFrame(records, columns=["road", "slk_from", "slk_to", "value"])


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_overlaps_property_matches_legacy(seed: int):
    rng = random.Random(seed)
    target = _generate_segments(rng, group_count=3, per_group=20)
    data = _generate_data_segments(rng, group_count=3, count=30, max_len=1000)

    actions = [
        merge_module.Action(
            "value",
            rename="avg",
            aggregation=merge_module.Aggregation.LengthWeightedAverage(),
        ),
        merge_module.Action(
            "value",
            rename="sum",
            aggregation=merge_module.Aggregation.Sum(),
        ),
    ]

    legacy = merge_module.on_slk_intervals(
        target=target,
        data=data,
        join_left=["road"],
        column_actions=actions,
        from_to=("slk_from", "slk_to"),
    )

    optimized = merge_module.on_slk_intervals_optimized(
        target=target,
        data=data,
        join_left=["road"],
        column_actions=actions,
        from_to=("slk_from", "slk_to"),
    )

    pd.testing.assert_frame_equal(
        legacy.sort_index(axis=1),
        optimized.sort_index(axis=1),
        check_dtype=False,
    )
