



import pandas as pd


def test_recombine_segmentation_index():
    from pyroads.segmenter._util.split_rows_by_category_to_max_measure_length import _recombine_segmentation_index


    original_data = pd.DataFrame(
        columns = ["from","to","value"],
        data=[
            [ 0,15,"a"],
            [15,24,"b"],
            [24,30,"c"],
            [35,50,"d"],
        ]
    )

    segmentation = pd.DataFrame(
        columns = [
            "from",
            "to",
            "segment_index",
            "index_from",
            "index_to"
        ],
        data = [
            [ 0,10,1,0,1],
            [10,15,1,0,1],
            [15,20,1,1,2],
            [20,30,1,1,2],
            [35,45,2,3,4],
        ]
    )

    result = segmentation.copy();
    result["recombination_index"] = _recombine_segmentation_index(
        segmentation   = segmentation,
        original_data  = original_data,
        measure        = ("from", "to"),
        grouping_id    = "segment_index",
        grouping_range = ("index_from", "index_to")
    )

    expected_result = pd.DataFrame(
        columns = [
            "from",
            "to",
            "segment_index",
            "index_from",
            "index_to",
            "recombination_index"
        ],
        data = [
            [ 0, 10, 1, 0, 1, 0],
            [10, 15, 1, 0, 1, 0],
            [15, 20, 1, 1, 2, 1],
            [20, 30, 1, 1, 2, 1],
            [35, 45, 2, 3, 4, 3],
        ]
    )

    pd.testing.assert_frame_equal(
        result,
        expected_result,
        check_like=False # ignore column and row order
    )

