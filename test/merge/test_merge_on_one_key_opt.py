import pandas as pd
import pyroads.merge.merge as merge


def test_merge_on_one_key():
    segments = pd.DataFrame(
        columns=["road", "slk_from", "slk_to"],
        data=[
            ["H001", 0, 100],
            ["H001", 100, 200],
            ["H001", 200, 300],
            ["H001", 300, 400],
        ],
    )

    data = pd.DataFrame(
        columns=["road", "slk_from", "slk_to", "measure", "category"],
        data=[
            ["H001", 50, 140, 1.0, "A"],  # 50  40   0  0
            ["H001", 140, 160, 2.0, "B"],  # 0  20   0  0
            ["H001", 160, 180, 3.0, "B"],  # 0  20   0  0
            ["H001", 180, 220, 4.0, "B"],  # 0  20  20  0
            ["H001", 220, 240, 5.0, "C"],  # 0   0  20  0
            ["H001", 240, 260, 5.0, "C"],  # 0   0  20  0
            ["H001", 260, 280, 6.0, "D"],  # 0   0  20  0
            ["H001", 280, 300, 7.0, "E"],  # 0   0  20  0
            ["H001", 300, 320, 8.0, "F"],  # 0   0     20
        ],
    )

    expected_output = pd.DataFrame(
        columns=[
            "road",
            "slk_from",
            "slk_to",
            "measure longest value",
            "category longest value",
        ],
        data=[
            ["H001", 0, 100, 1.0, "A"],
            ["H001", 100, 200, 1.0, "B"],
            ["H001", 200, 300, 5.0, "C"],
            ["H001", 300, 400, 8.0, "F"],
        ],
    )

    res = merge.on_slk_intervals_optimized(
        segments,
        data,
        ["road"],
        [
            merge.Action(
                "measure",
                rename="measure longest value",
                aggregation=merge.Aggregation.KeepLongest(),
            ),
            merge.Action(
                "category",
                rename="category longest value",
                aggregation=merge.Aggregation.KeepLongest(),
            ),
        ],
        from_to=("slk_from", "slk_to"),
    )
    assert res.compare(expected_output).empty
