from .lane_to_col import lane_to_col as lane_to_col
from .lane_to_row import lane_to_row as lane_to_row
from .lane_side_split import lane_side_split as lane_side_split
from .._dataframe import supports_pandas_and_polars

lane_to_col = supports_pandas_and_polars(lane_to_col)
lane_to_row = supports_pandas_and_polars(lane_to_row)
lane_side_split = supports_pandas_and_polars(lane_side_split)