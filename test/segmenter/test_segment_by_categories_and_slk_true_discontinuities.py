import pandas as pd
import numpy as np


def test_segment_by_categories_and_slk_true_discontinuities():
    from pyroads.segmenter import segment_by_categories_and_slk_true_discontinuities
    
    data_to_segment = pd.DataFrame(
        columns=["road_no", "carriageway", "xsp", "slk_from", "slk_to", "true_from", "true_to", "value"],
        data=[
            ["H001", "L", "L1", 0.010, 0.050, 0.010, 0.050, "a"], # SLK discontinuity (gap)
            ["H001", "L", "L1", 0.050, 0.070, 0.050, 0.070, "b"],
            ["H001", "L", "L1", 0.080, 0.100, 0.070, 0.090, "c"],
            ["H001", "L", "L1", 0.100, 0.120, 0.090, 0.110, "d"],

            ["H001", "L", "L2", 0.010, 0.050, 0.010, 0.050, "a"], # TRUE discontinuity (gap, overlaps not permitted for true)
            ["H001", "L", "L2", 0.050, 0.070, 0.050, 0.070, "b"],
            ["H001", "L", "L2", 0.070, 0.080, 0.070, 0.080, "c"],
            ["H001", "L", "L2", 0.080, 0.090, 0.090, 0.110, "d"],
            ["H001", "L", "L2", 0.090, 0.100, 0.110, 0.120, "e"],

            ["H001", "L", "L3", 0.080, 0.100, 0.070, 0.090, "e"], # XSP discontinuity
            ["H001", "L", "L3", 0.100, 0.120, 0.090, 0.110, "f"],

            ["H001", "R", "L3", 0.010, 0.050, 0.010, 0.050, "a"], # SLK discontinuity (overlap)
            ["H001", "R", "L3", 0.050, 0.070, 0.050, 0.070, "b"],
            ["H001", "R", "L3", 0.050, 0.060, 0.070, 0.080, "c"],
            ["H001", "R", "L3", 0.060, 0.070, 0.080, 0.100, "d"],
            ["H001", "R", "L3", 0.070, 0.090, 0.100, 0.110, "e"],
        ]
    )

    actual_result = data_to_segment.copy()
    actual_result["seg.ctg"] = segment_by_categories_and_slk_true_discontinuities(
        data_to_segment,
        ["road_no", "carriageway", "xsp"],
        measure_slk =("slk_from", "slk_to"),
        measure_true=("true_from","true_to")
    )
    
    expected_result = pd.DataFrame(
        columns=["road_no", "carriageway", "xsp", "slk_from", "slk_to", "true_from", "true_to", "value" , "seg.ctg"],
        data=[
            ["H001", "L", "L1", 0.010, 0.050, 0.010, 0.050, "a", 0], # SLK discontinuity (gap)
            ["H001", "L", "L1", 0.050, 0.070, 0.050, 0.070, "b", 0],
            ["H001", "L", "L1", 0.080, 0.100, 0.070, 0.090, "c", 1],
            ["H001", "L", "L1", 0.100, 0.120, 0.090, 0.110, "d", 1],

            ["H001", "L", "L2", 0.010, 0.050, 0.010, 0.050, "a", 2], # TRUE discontinuity (gap, overlaps not permitted for true)
            ["H001", "L", "L2", 0.050, 0.070, 0.050, 0.070, "b", 2],
            ["H001", "L", "L2", 0.070, 0.080, 0.070, 0.080, "c", 2],
            ["H001", "L", "L2", 0.080, 0.090, 0.090, 0.110, "d", 3],
            ["H001", "L", "L2", 0.090, 0.100, 0.110, 0.120, "e", 3],

            ["H001", "L", "L3", 0.080, 0.100, 0.070, 0.090, "e", 4], # XSP discontinuity
            ["H001", "L", "L3", 0.100, 0.120, 0.090, 0.110, "f", 4],

            ["H001", "R", "L3", 0.010, 0.050, 0.010, 0.050, "a", 5], # SLK discontinuity (overlap)
            ["H001", "R", "L3", 0.050, 0.070, 0.050, 0.070, "b", 5],
            ["H001", "R", "L3", 0.050, 0.060, 0.070, 0.080, "c", 6],
            ["H001", "R", "L3", 0.060, 0.070, 0.080, 0.100, "d", 6],
            ["H001", "R", "L3", 0.070, 0.090, 0.100, 0.110, "e", 6],
        ]
    )

    assert actual_result.compare(expected_result).empty



    