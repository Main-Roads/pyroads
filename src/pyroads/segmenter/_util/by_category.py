


from typing import List, Tuple
import pandas
import numpy as np
CATEGORY_COLUMN_NAME = "seg.ctg"



def segment_by_categories_and_slk_discontinuities(
        data:pandas.DataFrame,
        categories:List[str],
        measure_slk:Tuple[str,str],
    ):
    """
    Returns a series containing integer segment labels:
    A new 'segment' is started whenever one of the `categories` changes or
    any time there is a discontinuity in the slk measure.
    
    
    Please Note:
    
    - For each unique combination of `categories` in the input `data`, the range `slk_from` to `slk_to` for observations **must** be non-overlapping.
    - No check is made for overlapping observations in this function, it is actually very computationally intensive to do so. I might write a utility that does it one day.
    - Weird stuff will happen if there are overlaps
    - You have been warned

    Internally, data is sorted by the `categories` (in order provided) then by 
    `measure_slk[0]` prior to seeking discontinuities then labeling.

    Args:
        data (pandas.DataFrame):       data to be segmented
        categories (list[str]):        column names of categories to segment by; eg ["road", "cwy"] or ["road", "cwy", "xsp"]
        measure_slk (tuple[str,str]):  column names of slk measure to segment by; eg ("slk_from", "slk_to")
    Returns:
        pandas.Series: A series of integers which label the segment_id of each row.
        A series with an index that is compatible
        with the input `data` such that it can be easily joined or assigned to the original dataframe.
        See example below for suggested usage.
        

    Example:
    
    ```python
    df["segment_id"] = segment_by_categories_and_slk_true_discontinuities(
        data         = df,
        categories   = ["road", "cwy", "xsp"],
        measure_slk  = ("slk_from", "slk_to"),
        measure_true = ("true_from", "true_to")
    )
    ```
    """
    
    measure_from, measure_to = measure_slk
    data = data.sort_values(by=[*categories, measure_from])
    category_values = data[categories]
    boundaries = category_values.ne(category_values.shift()).any(axis=1).to_numpy(copy=True)
    boundaries[0] = True
    discontinuities = (
        np.around(data[measure_to].to_numpy()[:-1], 3)
        != np.around(data[measure_from].to_numpy()[1:], 3)
    )
    boundaries[1:] |= discontinuities
    segment_ids = np.cumsum(boundaries, dtype=np.int64) - 1
    return pandas.Series(segment_ids.astype("u8"), index=data.index)

def segment_by_categories_and_slk_true_discontinuities(
        data:pandas.DataFrame,
        categories:List[str],
        measure_slk:Tuple[str,str],
        measure_true:Tuple[str,str]
    ) -> pandas.Series:
    """
    Returns a series containing integer segment labels:
    A new 'segment' is started whenever one of the `categories` changes or
    any time there is a discontinuity in the slk and/or true measure.
    
    
    Please Note:
    
    - For each unique combination of `categories` in the input `data`, the range `true_from` to `true_to` for observations **must** be non-overlapping.
    - No check is made for overlapping observations in this function, it is actually very computationally intensive to do so. I might write a utility that does it one day.
    - Weird stuff will happen if there are overlaps
    - You have been warned

    Internally, data is sorted by the `categories` (in order provided) then by 
    `measure_true[0]` prior to seeking discontinuities then labeling.

    Args:
        data (pandas.DataFrame):       data to be segmented
        categories (list[str]):        column names of categories to segment by; eg ["road", "cwy"] or ["road", "cwy", "xsp"]
        measure_slk (tuple[str,str]):  column names of slk measure to segment by; eg ("slk_from", "slk_to")
        measure_true (tuple[str,str]): column names of true measure to segment by; eg ("true_from", "true_to")
    Returns:
        pandas.Series: A series of integers which label the segment_id of each row.
        A series with an index that is compatible
        with the input `data` such that it can be easily joined or assigned to the original dataframe.
        See example below for suggested usage.
        

    Example:
    
    ```python
    df["segment_id"] = segment_by_categories_and_slk_true_discontinuities(
        data         = df,
        categories   = ["road", "cwy", "xsp"],
        measure_slk  = ("slk_from", "slk_to"),
        measure_true = ("true_from", "true_to")
    )
    ```

    """
    
    measure_slk_from, measure_slk_to = measure_slk
    measure_true_from, measure_true_to = measure_true

    # Sorting once makes category boundaries and adjacent interval comparisons
    # available as contiguous NumPy arrays, avoiding a Python loop per group.
    data = data.sort_values(by=[*categories, measure_true_from])
    category_values = data[categories]
    category_boundaries = category_values.ne(category_values.shift()).any(axis=1).to_numpy(copy=True)
    category_boundaries[0] = True

    slk_discontinuities = (
        np.around(data[measure_slk_to].to_numpy()[:-1], 3)
        != np.around(data[measure_slk_from].to_numpy()[1:], 3)
    )
    true_discontinuities = (
        np.around(data[measure_true_to].to_numpy()[:-1], 3)
        != np.around(data[measure_true_from].to_numpy()[1:], 3)
    )

    boundaries = category_boundaries.copy()
    boundaries[1:] |= slk_discontinuities | true_discontinuities
    segment_ids = np.cumsum(boundaries, dtype=np.int64) - 1

    return pandas.Series(segment_ids.astype("u8"), index=data.index)



def segment_by_cross_section(
        data,
        categories:List[str],
        lane_category:str,
        measure_slk:Tuple[str,str],
        measure_true:Tuple[str,str]
    ):
    """
    Returns a series containing integer segment labels:
    A new 'segment' is started whenever one of the `categories` changes or
    """
    data[segmentation_id]
