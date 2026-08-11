import pandas as pd
import pytest
import pyroads.merge.merge as merge


def test_keep_longest():
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
            ["H001", "L", 50, 140, 1.0, "A"],  # 50  40   0  0
            ["H001", "L", 140, 160, 2.0, "B"],  # 0  20   0  0
            ["H001", "L", 160, 180, 3.0, "B"],  # 0  20   0  0
            ["H001", "L", 180, 220, 4.0, "B"],  # 0  20  20  0
            ["H001", "L", 220, 240, 5.0, "C"],  # 0   0  20  0
            ["H001", "L", 240, 260, 5.0, "C"],  # 0   0  20  0
            ["H001", "L", 260, 280, 6.0, "D"],  # 0   0  20  0
            ["H001", "L", 280, 300, 7.0, "E"],  # 0   0  20  0
            ["H001", "L", 300, 320, 8.0, "F"],  # 0   0     20
        ],
    )

    expected_output = pd.DataFrame(
        columns=[
            "road",
            "cwy",
            "slk_from",
            "slk_to",
            "measure longest segment",
            "measure longest value",
            "category longest segment",
            "category longest value",
        ],
        data=[
            ["H001", "L", 0, 100, 1.0, 1.0, "A", "A"],
            ["H001", "L", 100, 200, 1.0, 1.0, "A", "B"],
            ["H001", "L", 200, 300, 4.0, 5.0, "B", "C"],
            ["H001", "L", 300, 400, 8.0, 8.0, "F", "F"],
        ],
    )

    with pytest.deprecated_call():
        deprecated_keep_longest_segment = merge.Aggregation.KeepLongestSegment()

    res = merge.on_slk_intervals_optimized(
        segments,
        data,
        ["road", "cwy"],
        [
            merge.Action(
                "measure",
                rename="measure longest segment",
                aggregation=deprecated_keep_longest_segment,
            ),
            merge.Action(
                "measure",
                rename="measure longest value",
                aggregation=merge.Aggregation.KeepLongest(),
            ),
            merge.Action(
                "category",
                rename="category longest segment",
                aggregation=deprecated_keep_longest_segment,
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
