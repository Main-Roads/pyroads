

import pytest


def test_check_linear_index():
    import pandas as pd
    import numpy as np
    from pyroads.segmenter._util.check_segmentation import check_linear_index

    # passing test
    df = pd.DataFrame(
        columns=["slk_from", "slk_to"],
        data=[
            [0,1],
            [1,2],
        ],
    )
    check_linear_index(df)


    with pytest.raises(ValueError, match="exactly two"):
        df = pd.DataFrame(
            columns=["slk_from", "slk_to","slk"],
            data=[
                [0,1,0],
                [1,2,0],
            ],
        )
        check_linear_index(df)
    
    with pytest.raises(TypeError, match="not a numeric"):
        df = pd.DataFrame(
            columns=["slk_from", "slk_to"],
            data=[
                ["0",1.0],
                ["1",2.0],
            ],
        )
        check_linear_index(df)

    with pytest.raises(TypeError, match="not a numeric"):
        df = pd.DataFrame(
            columns=["slk_from", "slk_to"],
            data=[
                [0,"1.0"],
                [1,2.0],
            ],
        )
        check_linear_index(df)
    
    with pytest.raises(ValueError, match="NaN"):
        df = pd.DataFrame(
            columns=["slk_from", "slk_to"],
            data=[
                [0,1],
                [1,np.nan],
            ],
        )
        check_linear_index(df)

    with pytest.raises(ValueError, match="first column is greater"):
        df = pd.DataFrame(
            columns=["slk_from", "slk_to"],
            data=[
                [0,1],
                [3,1],
            ],
        )
        check_linear_index(df)

    with pytest.raises(ValueError, match="zero length"):
        df = pd.DataFrame(
            columns=["slk_from", "slk_to"],
            data=[
                [0,1],
                [3,3],
            ],
        )
        check_linear_index(df)

def test_check_linear_index_is_ordered_and_disjoint():
    import pandas as pd
    import numpy as np
    from pyroads.segmenter._util.check_segmentation import check_linear_index, check_linear_index_is_ordered_and_disjoint
    
    # passing test
    df = pd.DataFrame(
        columns=["group_categories","slk_from", "slk_to"],
        data=[
            ["a",1,2],
            ["a",2,3],
            ["b",5,6],
            ["b",6,7],
        ],
    )
    # confirm passes regular test
    check_linear_index(df[["slk_from", "slk_to"]])
    # confirm fails disjoint test
    check_linear_index_is_ordered_and_disjoint(df, measure=("slk_from", "slk_to"),categories=["group_categories"])



    with pytest.raises(ValueError, match="monotonic increasing"):
        df = pd.DataFrame(
            columns=["group_categories","slk_from", "slk_to"],
            data=[
                ["a",1,2],
                ["a",0,3],
                ["b",5,6],
                ["b",6,7],
            ],
        )
        # confirm passes regular test
        check_linear_index(df[["slk_from", "slk_to"]])
        # confirm fails disjoint test
        check_linear_index_is_ordered_and_disjoint(df, measure=("slk_from", "slk_to"),categories=["group_categories"])
    
    
    with pytest.raises(ValueError, match="monotonic increasing"):
        df = pd.DataFrame(
            columns=["group_categories","slk_from", "slk_to"],
            data=[
                ["a",1,2],
                ["a",2,3],
                ["b",5,7],
                ["b",6,6.5],
            ],
        )
        # confirm passes regular test
        check_linear_index(df[["slk_from", "slk_to"]])
        # confirm fails disjoint test
        check_linear_index_is_ordered_and_disjoint(df, measure=("slk_from", "slk_to"),categories=["group_categories"])
    
    with pytest.raises(ValueError, match="not disjoint"):
        df = pd.DataFrame(
            columns=["group_categories","slk_from", "slk_to"],
            data=[
                ["a",1,3],
                ["a",3,5],
                ["b",5,7],
                ["b",6,8],
            ],
        )
        # confirm passes regular test
        check_linear_index(df[["slk_from", "slk_to"]])
        # confirm fails disjoint test
        check_linear_index_is_ordered_and_disjoint(df, measure=("slk_from", "slk_to"),categories=["group_categories"])