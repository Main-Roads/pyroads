import pandas as pd

from pyroads.merge import merge as merge


def _categorical_tie_frames():
    target = pd.DataFrame(
        columns=["road", "slk_from", "slk_to"],
        data=[["H001", 0, 30]],
    )
    data = pd.DataFrame(
        columns=["road", "slk_from", "slk_to", "category"],
        data=[
            ["H001", 0, 10, "A"],
            ["H001", 10, 20, "B"],
            ["H001", 20, 25, "A"],
            ["H001", 25, 30, "B"],
        ],
    )
    actions = [
        merge.Action(
            "category",
            rename="category_keep",
            aggregation=merge.Aggregation.KeepLongest(),
        )
    ]
    return target, data, actions


def _categorical_nonlexical_tie_frames():
    target = pd.DataFrame(
        columns=["road", "slk_from", "slk_to"],
        data=[["H001", 0, 30]],
    )
    data = pd.DataFrame(
        columns=["road", "slk_from", "slk_to", "category"],
        data=[
            ["H001", 0, 10, "C"],
            ["H001", 10, 20, "A"],
            ["H001", 20, 25, "C"],
            ["H001", 25, 30, "A"],
        ],
    )
    actions = [
        merge.Action(
            "category",
            rename="category_keep",
            aggregation=merge.Aggregation.KeepLongest(),
        )
    ]
    return target, data, actions


def test_keep_longest_fallback_tie_prefers_first_legacy():
    target, data, actions = _categorical_tie_frames()
    result = merge.on_slk_intervals(
        target=target,
        data=data,
        join_left=["road"],
        column_actions=actions,
        from_to=("slk_from", "slk_to"),
    )
    assert result.loc[0, "category_keep"] == "A"


def test_keep_longest_fallback_tie_prefers_first_optimized():
    target, data, actions = _categorical_tie_frames()
    result = merge.on_slk_intervals_optimized(
        target=target,
        data=data,
        join_left=["road"],
        column_actions=actions,
        from_to=("slk_from", "slk_to"),
    )
    assert result.loc[0, "category_keep"] == "A"


def test_keep_longest_nonlexical_tie_prefers_first_legacy():
    target, data, actions = _categorical_nonlexical_tie_frames()
    result = merge.on_slk_intervals(
        target=target,
        data=data,
        join_left=["road"],
        column_actions=actions,
        from_to=("slk_from", "slk_to"),
    )
    assert result.loc[0, "category_keep"] == "C"


def test_keep_longest_nonlexical_tie_prefers_first_optimized():
    target, data, actions = _categorical_nonlexical_tie_frames()
    result = merge.on_slk_intervals_optimized(
        target=target,
        data=data,
        join_left=["road"],
        column_actions=actions,
        from_to=("slk_from", "slk_to"),
    )
    assert result.loc[0, "category_keep"] == "C"
