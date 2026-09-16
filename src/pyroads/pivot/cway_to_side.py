import pandas as pd
from .._dataframe import supports_pandas_and_polars

@supports_pandas_and_polars
def cway_to_side(data, cway):

	data_right = data[data[cway].isin(['s', 'r', 'S', 'R'])].copy()
	data_right.loc[:,'side'] = 'R'
	data_left = data[data[cway].isin(['s', 'l', 'S', 'R'])].copy()
	data_left.loc[:,'side'] = 'L'
	new_data = pd.concat([data_right, data_left])
	return new_data.reset_index(drop = True)
