from io import StringIO
import pandas as pd
import numpy as np
from pyroads.homogeneous_segmentation import segment_ids_to_maximize_spatial_heterogeneity
from pyroads.homogeneous_segmentation._seg_shs import cumulative_q
import pytest


df1 = pd.read_csv(StringIO(""""var_a","var_b","slk_from","slk_to"
1,3,0,0.01
2,2,0.01,0.02
3,1,0.02,0.03
4,3,0.03,0.04
2,5,0.04,0.05
3,6,0.05,0.06
1,5,0.06,0.07
5,6,0.07,0.08
3,4,0.08,0.09
4,7,0.09,0.1
6,9,0.1,0.11
4,4,0.11,0.12
3,3,0.12,0.13
7,2,0.13,0.14
5,3,0.14,0.15
6,1,0.15,0.16
4,3,0.16,0.17
5,2,0.17,0.18
"""))


def test_shs_df1():
    result = df1.copy()
    result["seg.id"] = segment_ids_to_maximize_spatial_heterogeneity(
        data                         = result,
        measure                      = ("slk_from", "slk_to"),
        variable_column_names                    = ["var_a","var_b"],
        allowed_segment_length_range = (0.020, 0.100),
    )
    expected_output = pd.read_csv(StringIO(""""var_a","var_b","slk_from","slk_to","seg.id"
        1,3,0,0.01,1
        2,2,0.01,0.02,1
        3,1,0.02,0.03,1
        4,3,0.03,0.04,1
        2,5,0.04,0.05,1
        3,6,0.05,0.06,1
        1,5,0.06,0.07,1
        5,6,0.07,0.08,2
        3,4,0.08,0.09,2
        4,7,0.09,0.1,2
        6,9,0.1,0.11,2
        4,4,0.11,0.12,2
        3,3,0.12,0.13,2
        7,2,0.13,0.14,3
        5,3,0.14,0.15,3
        6,1,0.15,0.16,3
        4,3,0.16,0.17,3
        5,2,0.17,0.18,3
    """))
    pd.testing.assert_frame_equal(
        left=expected_output,
        right=result
    )


def test_shs_df2():
    df2 = pd.read_csv("./test/homogeneous_segmentation/test_data/df2.csv")

    df2 = df2[
          (df2["road"]=="H001")
        & (df2["cwy"]=="S")
        & (df2["dirn"]=="L")
        & (df2["slk_from"]>50)
        & (df2["slk_from"]<60)
    ]

    py_output = df2
    py_output["seg.id"] = segment_ids_to_maximize_spatial_heterogeneity(
        data                         = df2,
        measure                      = ("slk_from", "slk_to"),
        variable_column_names                    = ["deflection"],
        allowed_segment_length_range = (0.050, 0.200)
    )
    py_output = py_output.sort_values(by="slk_from").reset_index(drop=True)
    
    r_output = (
        pd.read_csv("./test/homogeneous_segmentation/r_outputs/df2_seg_test_out.csv")
        .reset_index(drop=True)
        .sort_values(by="slk_from")
    )

    pd.testing.assert_frame_equal(
        left=r_output,
        right=py_output
    )

@pytest.mark.parametrize("data", [df1["var_a"].values, df1["var_b"].values])
def test_q_cumulative(data):
    result = cumulative_q(data)
    result_slow = []
    divisor = np.var(data) * len(data)
    for index in range(1, len(data)):
        da = data[:index]
        db = data[index:]
        result_slow.append(
            1 - (len(da) * np.var(da) + len(db)* np.var(db)) / divisor
        )
    assert all((np.array(result_slow) - result) < 0.0000001)
    