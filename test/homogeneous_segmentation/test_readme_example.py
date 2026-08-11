"""Tests for code snippets used in the consolidated homogeneous segmentation docs.
Tests are written so the body of each function can be copied into the docs.
therefore imports are made inside the functions and not at the top of the file.
"""
# pylint: disable=missing-function-docstring, import-outside-toplevel

def test_readme_example():
    from pyroads.homogeneous_segmentation import (
        segment_ids_to_maximize_spatial_heterogeneity,
        segment_ids_to_minimize_coefficient_of_variation
    )
    import pandas as pd
    from io import StringIO

    df = pd.read_csv(StringIO("""road,slk_from,slk_to,cwy,deflection,dirn
    H001,0.00,0.01,L,179.37,L
    H001,0.01,0.02,L,177.12,L
    H001,0.02,0.03,L,179.06,L
    H001,0.03,0.04,L,212.65,L
    H001,0.04,0.05,L,175.35,L
    H001,0.05,0.06,L,188.66,L
    H001,0.06,0.07,L,188.31,L
    H001,0.07,0.08,L,174.48,L
    H001,0.08,0.09,L,210.28,L
    H001,0.09,0.10,L,260.05,L
    H001,0.10,0.11,L,228.83,L
    H001,0.11,0.12,L,226.33,L
    H001,0.12,0.13,L,245.53,L
    H001,0.13,0.14,L,315.77,L
    H001,0.14,0.15,L,373.86,L
    H001,0.15,0.16,L,333.56,L"""))

    df["seg.shs"] = segment_ids_to_maximize_spatial_heterogeneity(
        data                         = df,
        measure                      = ("slk_from", "slk_to"),
        variable_column_names        = ["deflection"],
        allowed_segment_length_range = (0.030, 0.080)
    )

    df["seg.mcv"] = segment_ids_to_minimize_coefficient_of_variation(
        data                         = df,
        measure                      = ("slk_from", "slk_to"),
        variable_column_names        = ["deflection"],
        allowed_segment_length_range = (0.030, 0.080)
    )


    expected_result          = pd.read_csv(StringIO("""road,slk_from,slk_to,cwy,deflection,dirn,seg.shs,seg.mcv
    H001,0.00,0.01,L,179.37,L,1,1
    H001,0.01,0.02,L,177.12,L,1,1
    H001,0.02,0.03,L,179.06,L,1,1
    H001,0.03,0.04,L,212.65,L,1,1
    H001,0.04,0.05,L,175.35,L,2,2
    H001,0.05,0.06,L,188.66,L,2,2
    H001,0.06,0.07,L,188.31,L,2,2
    H001,0.07,0.08,L,174.48,L,2,2
    H001,0.08,0.09,L,210.28,L,2,2
    H001,0.09,0.10,L,260.05,L,3,3
    H001,0.10,0.11,L,228.83,L,3,3
    H001,0.11,0.12,L,226.33,L,3,3
    H001,0.12,0.13,L,245.53,L,3,3
    H001,0.13,0.14,L,315.77,L,3,3
    H001,0.14,0.15,L,373.86,L,3,3
    H001,0.15,0.16,L,333.56,L,3,3
    """))

    # check the result matches the expected result
    pd.testing.assert_frame_equal(
        left       = df,
        right      = expected_result,
        check_like = True # ignore column and row order
    )
