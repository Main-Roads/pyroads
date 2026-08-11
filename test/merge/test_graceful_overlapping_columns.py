import pandas as pd
import pytest
import re


def test_graceful_overlapping_columns_error():
    import pyroads.merge.merge as merge

    segments = pd.DataFrame(
        columns=["road", "cwy", "slk_from", "slk_to", "measure"],
        data=[
            ["H001", "L", 0, 100, 0],
            ["H001", "L", 100, 200, 0],
            ["H001", "L", 200, 300, 0],
            ["H001", "L", 300, 400, 0],
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

    with pytest.raises(
        Exception,
        match=re.escape(
            "Cannot merge column 'measure' into target because the target already contains a column of that name. Please consider using the rename parameter; `Action(..., rename='xyz')`"
        ),
    ):
        merge.on_slk_intervals(
            target=segments,
            data=data,
            join_left=["road", "cwy"],
            column_actions=[
                merge.Action("measure", aggregation=merge.Aggregation.KeepLongest()),
            ],
            from_to=("slk_from", "slk_to"),
        )

    with pytest.raises(
        Exception,
        match=re.escape(
            "Cannot merge column 'category' as 'measure' into target because the target already contains a column named 'measure'."
        ),
    ):
        merge.on_slk_intervals(
            target=segments,
            data=data,
            join_left=["road", "cwy"],
            column_actions=[
                merge.Action(
                    "category",
                    rename="measure",
                    aggregation=merge.Aggregation.KeepLongest(),
                ),
            ],
            from_to=("slk_from", "slk_to"),
        )

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
