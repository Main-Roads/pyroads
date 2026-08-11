import pandas as pd

from pyroads.merge import Action, Aggregation, on_slk_intervals


def test_interval_merge():
    segments = pd.DataFrame(
        [
            ["H001", 0, 100],
            ["H001", 100, 200],
            ["H001", 200, 300],
            ["H001", 300, 400],
        ],
        columns=["road", "slk_from", "slk_to"],
    )
    data = pd.DataFrame(
        [
            ["H001", 50, 140, 1.0, "A"],
            ["H001", 140, 160, 2.0, "B"],
            ["H001", 160, 180, 3.0, "B"],
            ["H001", 180, 220, 4.0, "B"],
            ["H001", 220, 240, 5.0, "C"],
            ["H001", 240, 260, 5.0, "C"],
            ["H001", 260, 280, 6.0, "D"],
            ["H001", 280, 300, 7.0, "E"],
            ["H001", 300, 320, 8.0, "F"],
        ],
        columns=["road", "slk_from", "slk_to", "measure", "category"],
    )

    result = on_slk_intervals(
        target=segments,
        data=data,
        join_left=["road"],
        column_actions=[
            Action("measure", rename="measure longest value", aggregation=Aggregation.KeepLongest()),
            Action("category", rename="category longest value", aggregation=Aggregation.KeepLongest()),
        ],
        from_to=("slk_from", "slk_to"),
    )

    expected = pd.DataFrame(
        [
            ["H001", 0, 100, 1.0, "A"],
            ["H001", 100, 200, 1.0, "B"],
            ["H001", 200, 300, 5.0, "C"],
            ["H001", 300, 400, 8.0, "F"],
        ],
        columns=["road", "slk_from", "slk_to", "measure longest value", "category longest value"],
    )

    pd.testing.assert_frame_equal(result, expected)