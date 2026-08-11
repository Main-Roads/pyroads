from typing import Optional

import numpy as np
import pandas as pd

from .as_metres import as_metres

def make_segments(
	data: pd.DataFrame, 
	start: Optional[str] = None, 
	end: Optional[str] = None, 
	start_true: Optional[str] = None, 
	end_true: Optional[str] = None, 
	max_segment: int = 100, 
	split_ends: bool = True, 
	km: bool = True,
	id: bool = False,
	as_km: Optional[bool] = None,
	) -> 'pd.DataFrame':

	"""
	Takes larger segments of irregular lengths and splits them into a smaller regular length.
	The attributes of each original observation are repeated for each split output segment.
	End segments may be shorter than the specified length.

	Parameters:
	-----------
	data (DataFrame):  Dataframe containing the data to be reshaped.
	start (label):       Column name of the start of the segment.
	end (label):         Column name of the end of the segment.
	start_true (label):  Column name of the start of the segment, in true distance.
	end_true (label):    Column name of the end of the segment, in true distance.
	max_segment (int): Maximum segment length of output. Segments in the input that are already shorter may not be split.
	split_ends (bool): Prevent the last segment being very short by combining it with the second last segment then splitting that segment in half
	km (bool):         Start and End columns are measured in kilometres.

	Returns:
		(DataFrame) new dataframe with the same columns as the input, but with additional repeated rows such that the segmentation length now no longer than `max_segment`.
	
	Examples
	--------
	>>> df = pd.DataFrame({'ROAD': ['H001', 'H001', 'H001'], 
							'START_SLK': [12.20, 12.40, 12.6], 
							'END_SLK': [12.4, 12.56, 12.8]})
	
	>>> df
	    ROAD  START_SLK  END_SLK
	0   H001	12.2	 12.4
	1   H001	12.4	 12.56
	2   H001	12.6	 12.8

	Make 100m sections, split at the ends.

	>>> reshape.make_segments(df, start = 'START_SLK', end = 'END_SLK', max_segment=100)
	    ROAD  START_SLK  END_SLK Length
	0   H001	12.2       12.3    0.1
	1   H001	12.3       12.4    0.1
	2   H001	12.4       12.48   0.08
	3   H001	12.48      12.56   0.08
	4   H001	12.6       12.7    0.1
	5   H001	12.7       12.8    0.1
	"""

	if as_km is not None:
		km = as_km

	starts = [var for var in [start, start_true] if bool(var)]
	ends = [var for var in [end, end_true] if bool(var)]
	
	new_data = data.copy()  # Copy of the dataset
	
	slks = [slk for slk in starts + ends if slk is not None]
	
	new_data = new_data.dropna(thresh=2)  # drop any row that does not contain at least two non-missing values.
	
	# Change SLK variables to 32 bit integers of metres to avoid the issue with calculations on floating numbers
	for slk in slks:
		new_data[slk] = as_metres(new_data[slk])
	
	new_data.insert(len(new_data.columns) - 1, 'Length', new_data[ends[0]] - new_data[starts[0]])
	
	# Reshape the data into size specified in 'max_segment'
	new_data = new_data.reindex(new_data.index.repeat(np.ceil((new_data[ends[0]] - new_data[starts[0]]) / max_segment)))  # reindex by the number of intervals of specified length between the start and the end.
	
	#A start_end is any observation that is too small to be split, hence it is both the start and the end of a segment.	
	new_data['start_end'] = np.where(new_data['Length'] <= max_segment, True, False)

	for start_, end_ in zip(starts, ends):
		# Increment the start rows by the segment size
		new_data[start_] = (new_data[start_] + new_data.groupby(level=0).cumcount() * max_segment)
		new_data[end_] = np.where(((new_data[start_].shift(-1) - new_data[start_]) == max_segment), new_data[start_].shift(-1), new_data[end_])


	#&  ((new_data[end_].shift(-1) - new_data[start_].shift(-1)) >= max_segment)
	# Split the last segment if it is too short
	# Check for minimum segment lengths
	if split_ends:
		for start_, end_ in zip(starts, ends):
			# where the difference between the `end` and `start` is less than the minimum segment size and isn't a start_end, subtract the difference from the `start` and set the same value as the previous `end`
			new_data['too_short'] = np.where(((new_data[end_] - new_data[start_]) < max_segment) & (new_data['start_end'] == False), True, False)		
			new_data[end_] = np.where(new_data['too_short'].shift(-1) == True, (new_data[end_].shift(-1) + new_data[start_]) / 2, new_data[end_])
			new_data[start_] = np.where(new_data['too_short'] == True, new_data[end_].shift(1), new_data[start_])	
			#Find the 2 decimal place ceiling and floor for start and end values respectively.
			new_data[start_] = np.round(new_data[start_]/10) * 10
			new_data[end_] = np.round(new_data[end_]/10) * 10
			# Drop the boolean columns
			new_data = new_data.drop(['start_end', 'too_short'], axis=1)

	if km:
		# Convert SLK variables back to km
		for slk in slks:
			new_data[slk] = new_data[slk] / 1000
	
	# recalculate length
	new_data = new_data.drop('Length', axis = 1) 
	new_data['Length'] = new_data[ends[0]] - new_data[starts[0]]
	
	if id:
		new_data['original_id'] = new_data.index + 1
		new_data['segment_id'] = [i+1 for i in range(len(new_data))]		
	new_data = new_data.reset_index(drop = True)
	
	return new_data