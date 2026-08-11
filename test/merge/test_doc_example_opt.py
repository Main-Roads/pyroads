import pandas as pd
import pyroads.merge.merge as merge


def test_doc_example():
    segmentation = pd.DataFrame(
        columns=["road_no", "carriageway", "slk_from", "slk_to"],
        data=[
            ["H001", "L", 10, 50],
            ["H001", "L", 50, 100],
            ["H001", "L", 100, 150],
        ],
    )
    segmentation_copy = segmentation.copy()

    pavement_data = pd.DataFrame(
        columns=[
            "road_no",
            "carriageway",
            "slk_from",
            "slk_to",
            "pavement_width",
            "pavement_type",
        ],
        data=[
            ["H001", "L", 00, 10, 3.10, "tA"],
            ["H001", "L", 10, 20, 4.00, "tA"],
            ["H001", "L", 20, 40, 3.50, "tA"],
            ["H001", "L", 40, 80, 3.80, "tC"],
            ["H001", "L", 80, 130, 3.10, "tC"],
            ["H001", "L", 130, 140, 3.00, "tB"],
        ],
    )
    pavement_data_copy = pavement_data.copy()

    result = merge.on_slk_intervals_optimized(
        target=segmentation,
        data=pavement_data,
        join_left=["road_no", "carriageway"],
        column_actions=[
            merge.Action("pavement_width", merge.Aggregation.LengthWeightedAverage()),
            merge.Action("pavement_type", merge.Aggregation.KeepLongest()),
        ],
        from_to=("slk_from", "slk_to"),
    )
    assert result.compare(
        pd.DataFrame(
            columns=[
                "road_no",
                "carriageway",
                "slk_from",
                "slk_to",
                "pavement_width",
                "pavement_type",
            ],
            data=[
                ["H001", "L", 10, 50, 3.700, "tA"],
                ["H001", "L", 50, 100, 3.520, "tC"],
                ["H001", "L", 100, 150, 3.075, "tC"],
            ],
        )
    ).empty

    assert pavement_data.compare(pavement_data_copy).empty
    assert pavement_data.index.equals(pavement_data_copy.index)
    assert segmentation.compare(segmentation_copy).empty
    assert segmentation.index.equals(segmentation_copy.index)
