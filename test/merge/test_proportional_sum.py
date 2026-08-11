import pandas as pd
import pyroads.merge.merge as merge
import pytest


def test_proportional_sum():
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
            ["H001", "L", 50, 140, 1.0, "A"],  # 50 40   0  0
            ["H001", "L", 140, 160, 2.0, "B"],  # 0  20   0  0
            ["H001", "L", 160, 180, 3.0, "B"],  # 0  20   0  0
            ["H001", "L", 180, 220, 4.0, "B"],  # 0  20  20  0
            ["H001", "L", 220, 240, 5.0, "C"],  # 0   0  20  0
            ["H001", "L", 240, 260, 5.0, "C"],  # 0   0  20  0
            ["H001", "L", 260, 280, 6.0, "D"],  # 0   0  20  0
            ["H001", "L", 280, 290, 7.0, "E"],  # 0   0  10  0
            ["H001", "L", 290, 320, 8.0, "F"],  # 0   0  10 20
        ],
    )

    expected_output = pd.DataFrame(
        columns=["road", "cwy", "slk_from", "slk_to", "measure"],
        data=[
            ["H001", "L", 0, 100, 1.0 * 50 / 90],
            ["H001", "L", 100, 200, 1.0 * 40 / 90 + 2.0 + 3.0 + 4.0 * 20 / 40],
            [
                "H001",
                "L",
                200,
                300,
                4.0 * 20 / 40 + 5.0 + +5.0 + 6.0 + 7.0 + 8.0 * 10 / 30,
            ],
            ["H001", "L", 300, 400, 8.0 * 20 / 30],
        ],
    )
    with pytest.deprecated_call():
        deprecated_aggregation_call = merge.Aggregation.ProportionalSum()
    res = merge.on_slk_intervals(
        segments,
        data,
        ["road", "cwy"],
        [
            merge.Action("measure", aggregation=deprecated_aggregation_call),
        ],
        from_to=("slk_from", "slk_to"),
    )

    assert res.compare(expected_output).empty


def test_sum_proportion_of_data():
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
            ["H001", "L", 50, 140, 1.0, "A"],  # 50 40   0  0
            ["H001", "L", 140, 160, 2.0, "B"],  # 0  20   0  0
            ["H001", "L", 160, 180, 3.0, "B"],  # 0  20   0  0
            ["H001", "L", 180, 220, 4.0, "B"],  # 0  20  20  0
            ["H001", "L", 220, 240, 5.0, "C"],  # 0   0  20  0
            ["H001", "L", 240, 260, 5.0, "C"],  # 0   0  20  0
            ["H001", "L", 260, 280, 6.0, "D"],  # 0   0  20  0
            ["H001", "L", 280, 290, 7.0, "E"],  # 0   0  10  0
            ["H001", "L", 290, 320, 8.0, "F"],  # 0   0  10 20
        ],
    )

    expected_output = pd.DataFrame(
        columns=["road", "cwy", "slk_from", "slk_to", "measure"],
        data=[
            ["H001", "L", 0, 100, 1.0 * 50 / 90],
            [
                "H001",
                "L",
                100,
                200,
                1.0 * 40 / 90 + 2.0 * 20 / 20 + 3.0 * 20 / 20 + 4.0 * 20 / 40,
            ],
            [
                "H001",
                "L",
                200,
                300,
                4.0 * 20 / 40
                + 5.0 * 20 / 20
                + 5.0 * 20 / 20
                + 6.0 * 20 / 20
                + 7.0 * 10 / 10
                + 8.0 * 10 / 30,
            ],
            ["H001", "L", 300, 400, 8.0 * 20 / 30],
        ],
    )

    res = merge.on_slk_intervals(
        segments,
        data,
        ["road", "cwy"],
        [
            merge.Action(
                "measure", aggregation=merge.Aggregation.SumProportionOfData()
            ),
        ],
        from_to=("slk_from", "slk_to"),
    )

    assert res.compare(expected_output).empty


def test_sum_proportion_of_target():
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
            ["H001", "L", 50, 140, 1.0, "A"],  # 50 40   0  0
            ["H001", "L", 140, 160, 2.0, "B"],  # 0  20   0  0
            ["H001", "L", 160, 180, 3.0, "B"],  # 0  20   0  0
            ["H001", "L", 180, 220, 4.0, "B"],  # 0  20  20  0
            ["H001", "L", 220, 240, 5.0, "C"],  # 0   0  20  0
            ["H001", "L", 240, 260, 5.0, "C"],  # 0   0  20  0
            ["H001", "L", 260, 280, 6.0, "D"],  # 0   0  20  0
            ["H001", "L", 280, 290, 7.0, "E"],  # 0   0  10  0
            ["H001", "L", 290, 320, 8.0, "F"],  # 0   0  10 20
        ],
    )

    expected_output = pd.DataFrame(
        columns=["road", "cwy", "slk_from", "slk_to", "measure"],
        data=[
            ["H001", "L", 0, 100, 1.0 * 50 / 100],
            [
                "H001",
                "L",
                100,
                200,
                1.0 * 40 / 100 + 2.0 * 20 / 100 + 3.0 * 20 / 100 + 4.0 * 20 / 100,
            ],
            [
                "H001",
                "L",
                200,
                300,
                4.0 * 20 / 100
                + 5.0 * 20 / 100
                + 5.0 * 20 / 100
                + 6.0 * 20 / 100
                + 7.0 * 10 / 100
                + 8.0 * 10 / 100,
            ],
            ["H001", "L", 300, 400, 8.0 * 20 / 100],
        ],
    )

    res = merge.on_slk_intervals(
        segments,
        data,
        ["road", "cwy"],
        [
            merge.Action(
                "measure", aggregation=merge.Aggregation.SumProportionOfTarget()
            ),
        ],
        from_to=("slk_from", "slk_to"),
    )

    assert res.compare(expected_output).empty
