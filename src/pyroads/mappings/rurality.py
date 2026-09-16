import numpy as np
import pandas as pd
from deprecated import deprecated

from .._dataframe import supports_pandas_and_polars


@deprecated(version="0.2.0", reason="to be reviewed before being used. Perhaps to be taken out of pyroads.")
@supports_pandas_and_polars
def rurality(data: pd.DataFrame, RA: int, road_no: str, SLK: str = "SLK"):
	"""Determine whether the section is metro or rural."""
	regionType = np.where((data[RA] == 7 & data[road_no].isin(['M045', 'H006', 'M026', 'M010'])) |
						  ((data[SLK].astype(int) >= 48.5) & data[road_no] == 'H005') |
						  ((data[SLK].astype(int) >= 48.5) & data[road_no] == 'H005') |
						  ((data[SLK].astype(int) >= 12.22) & data[road_no] == 'H052') |
						  ((data[SLK].astype(int) >= 35.5) & data[road_no] == 'H001') |
						  ((data[SLK].astype(int) >= 18.54) & data[road_no] == 'H009'), 'Rural', 'Metro')
	return regionType