

import pandas as pd
import numpy as np

def test_pandas_unstack():
    # not sure how pandas treats missing values in an unstack. Lets test quick
    # .unstack() keeps rows even when all values are nan
    # seems to be a "reshape" call in disguise which should not discard data

    df = pd.DataFrame(
        columns=["idx", "xsp","data1","data2"],
        data=[
            [        0,     0,      0,      3],
            [        1,     0,      1,      2],
            [        2,     1,      2,      1],
            [        3,     1,      3,      0],
            [        4,     2,      2, np.nan],
            [        5,     2,      3, np.nan],
        ],
    )

    df = df.set_index(["idx", "xsp"])

    
    
    expected_df_unstack_data2 = pd.DataFrame(
        columns=["idx",      0,      1,      2],
        data=[
            [        0,      3, np.nan, np.nan],
            [        1,      2, np.nan, np.nan],
            [        2, np.nan,      1, np.nan],
            [        3, np.nan,      0, np.nan],
            [        4, np.nan, np.nan, np.nan],
            [        5, np.nan, np.nan, np.nan],
        ],
    ).set_index("idx")

    expected_df_unstack_data2.columns = expected_df_unstack_data2.columns.astype("i8").rename("xsp")

    pd.testing.assert_frame_equal(
        df["data2"].unstack("xsp"),
        expected_df_unstack_data2,
    )

