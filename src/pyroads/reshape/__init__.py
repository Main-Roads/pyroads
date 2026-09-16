from ._functions.get_segments import get_segments as get_segments
from ._functions.interval_merge import interval_merge as interval_merge
from ._functions.make_segments import make_segments as make_segments
from ._functions.stretch import stretch as stretch
from ._functions.as_metres import as_metres as as_metres
from .._dataframe import supports_pandas_and_polars

get_segments = supports_pandas_and_polars(get_segments)
interval_merge = supports_pandas_and_polars(interval_merge)
make_segments = supports_pandas_and_polars(make_segments)
stretch = supports_pandas_and_polars(stretch)
as_metres = supports_pandas_and_polars(as_metres)