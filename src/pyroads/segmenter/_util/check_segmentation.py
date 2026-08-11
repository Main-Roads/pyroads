import pandas as pd
from typing import List, Tuple

def check_monotonically_increasing_segments(df, categories:List[str], measure:Tuple[str,str]):
    """
    Check that the segments are monotonically increasing;
    Every segment in each category must start at or after the end of the previous segment

    - Note: does not perform a sort. Input data must be pre-sorted by measure[0]
    """
    return all(
        (values[measure[0]] >= values[measure[1]].shift(1, fill_value=0)).all()
        for _index, values
        in df.groupby(categories)
    )

def check_no_reversed_segments(df, measure:Tuple[str,str]):
    """
    Check that the measure[0] <= measure[1] for every segment

    measure might be `("true_from", "true_to")` for example.
    """
    return (df[measure[0]]<=df[measure[1]]).all()


def check_linear_index(measure:pd.DataFrame) -> None:
    """
    Take a dataframe consisting of exactly two numerical columns,
    representing ("slk_from","slk_to") or ("true_from", "true_to")

    Raises and error if there is some problem with the index

    - Non Numeric
    - Reversed From/To
    - Zero Length
    - NaN

    > Note: see `check_linear_index_is_ordered_and_disjoint` for more checks

    ### Example

    ```python
    from pyroads.segmenter._util.check_segmentation import check_linear_index
    
    df = pd.read_excel(...)

    check_linear_index(df[[ "slk_from",  "slk_to"]])
    check_linear_index(df[["true_from", "true_to"]])
    ```
    """
    column_names = ', '.join(measure.columns.to_list())

    if len(measure.columns)!=2:
        raise ValueError(f"A measure must consist of exactly two columns; found ({column_names})")

    if not pd.api.types.is_numeric_dtype(measure.dtypes.iloc[0]):
        raise TypeError(f"The column {measure.columns[0]} is not a numeric dtype; {measure.dtypes.iloc[0]}")

    if not pd.api.types.is_numeric_dtype(measure.dtypes.iloc[1]):
        raise TypeError(f"The column {measure.columns[1]} is not a numeric dtype; {measure.dtypes.iloc[1]}")

    if measure.isna().sum().sum() > 0:
        raise ValueError(
            f"The columns ({column_names}) contain at least one NaN value"
        )

    if (measure.iloc[:,0] > measure.iloc[:,1]).any():
        raise ValueError(
            f"The columns ({column_names}) contain at least one row where the first column is greater than the second"
        )
    
    if (measure.iloc[:,0] == measure.iloc[:,1]).any():
        raise ValueError(
            f"The columns ({column_names}) contain at least one zero length segment"
        )


def check_linear_index_is_ordered_and_disjoint(df, measure:Tuple[str,str], categories:List[str]):
    """
    Takes
    - a dataframe `df` 
    - a tuple `measure` (the names of the from/to measure columns
    which define a linear index eg `("true_from","true_to")`)
    - A list of `categories` (column names) by which to group the data before applying the check;
    
    The function will group the dataframe by `categories` then check
    - both `measure` columns are monotonically increasing
    - the 'measure from' column is always less than the 'measure to' column
    """

    # for group_index, group in df.groupby(categories):
    # TODO: Triggers useless future warning: if len(categories) == 1 then group_index will yield a length 1 tuple instead of a string.
    # To suppress the warning we can do something yuck:
    for group_index, group in df.groupby(categories[0] if len(categories)==1 else categories):
        measure_columns = group[[*measure]]
        if not measure_columns.iloc[:,0].is_monotonic_increasing:
            raise ValueError(
                f"The column {measure[0]} is not monotonic increasing for the group {categories}:{group_index}. Please try sorting your dataset by {measure[0]}. Otherwise remove overlapping segments."
            )
            
        if not measure_columns.iloc[:,1].is_monotonic_increasing:
            raise ValueError(
                f"The column {measure[1]} is not monotonic increasing for the group {categories}:{group_index}. Please try sorting your dataset by {measure[0]}. Otherwise remove overlapping segments."
            )
        
        if not (measure_columns.iloc[:1,1].to_numpy() <= measure_columns.iloc[1:,0].to_numpy()).all():
            raise ValueError(
                f"The columns {measure} have rows which are not disjoint intervals for the group {categories}:{group_index}."
        )