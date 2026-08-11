import pandas as pd
import pytest
import re


def test_graceful_missing_mismatched_columns():
    import pyroads.merge.merge as merge

    segments = pd.DataFrame(
        columns=["road", "cwy", "slk_from", "slk_to"],
        data=[
            ["H001", "L", 0, 100],
            ["H001", "L", 100, 200],
            ["H001", "L", 200, 300],
            ["H001", "L", 300, 400],
        ],
    )

    data = pd.DataFrame(
        columns=["road", "cwy", "slk_from", "slk_to", "measure", "category"],
        data=[
            ["H001", "L", 50, 140, 1.0, "A"],
            ["H001", "L", 140, 160, 2.0, "B"],
            ["H001", "L", 160, 180, 3.0, "B"],
        ],
    )

    # control case: should pass
    merge.on_slk_intervals(
        target=segments,
        data=data,
        join_left=["road", "cwy"],
        column_actions=[
            merge.Action(
                "measure",
                rename="measure_1",
                aggregation=merge.Aggregation.KeepLongest(),
            ),
        ],
        from_to=("slk_from", "slk_to"),
    )

    with pytest.raises(
        Exception,
        match=re.escape("Column 'slk_fro' is missing from both `target` and `data`."),
    ):
        # corrupted parameters
        merge.on_slk_intervals(
            target=segments,
            data=data,
            join_left=["road", "cwy"],
            column_actions=[
                merge.Action("measure", aggregation=merge.Aggregation.KeepLongest()),
            ],
            from_to=("slk_fro", "slk_to"),
        )

    with pytest.raises(
        Exception, match=re.escape("Column 'slk_from' is missing from `data`.")
    ):
        # corrupted dataframe
        merge.on_slk_intervals(
            target=segments,
            data=data.rename(columns={"slk_from": "slk_fro"}),
            join_left=["road", "cwy"],
            column_actions=[
                merge.Action("measure", aggregation=merge.Aggregation.KeepLongest()),
            ],
            from_to=("slk_from", "slk_to"),
        )

    with pytest.raises(
        Exception,
        match=re.escape("Column 'roadd' is missing from both `target` and `data`."),
    ):
        # corrupted parameters
        merge.on_slk_intervals(
            target=segments,
            data=data,
            join_left=["roadd", "cwy"],
            column_actions=[
                merge.Action("measure", aggregation=merge.Aggregation.KeepLongest()),
            ],
            from_to=("slk_from", "slk_to"),
        )

    with pytest.raises(
        Exception, match=re.escape("Column 'road' is missing from `data`.")
    ):
        # corrupted dataframe
        merge.on_slk_intervals(
            target=segments,
            data=data.rename(columns={"road": "rod"}),
            join_left=["road", "cwy"],
            column_actions=[
                merge.Action("measure", aggregation=merge.Aggregation.KeepLongest()),
            ],
            from_to=("slk_from", "slk_to"),
        )
