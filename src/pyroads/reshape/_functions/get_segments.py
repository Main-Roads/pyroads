from typing import Any, Optional, Union
import numpy as np

from .stretch import stretch
from .as_metres import as_metres

def get_segments(
	data: Any, 
	id_vars: Union[str, list],
	SLK: Optional[str] = None, 
	true_SLK: Optional[str] = None, 
	start: Optional[str] = None, 
	end: Optional[str] = None, 
	start_true: Optional[str] = None, 
	end_true: Optional[str] = None, 
	split_at: Union[str, list, bool] = False, 
	summarise: Union[dict, bool] = True, 
	segment_size: Optional[int] = None,
	) -> Any:

	"""
	Aggregates observations into segments grouped by categorical variables.
	Optionally, the output can simply return IDs for future aggregation.

	One of the following combinations must be provided:

	* `SLK`
	* `true_SLK`
	* `SLK` and `true_SLK`
	* `start` and `end`
	* `start_true` and `end_true`
	* `start`, `end`, `start_true` and `end_true`

	Parameters:
	-----------
	data (DataFrame):  Dataframe containing the data to be reshaped
	id_vars (label or list of labels)  :		   Column(s) to use as identifier variables for the linear reference points.
	SLK (label):               Column name of the linear reference point.
	true_SLK (label):          Column name of the linear reference point, in true distance.
	start (label):       Column name of the start of the segment.
	end (label):         Column name of the end of the segment.
	start_true (label):  Column name of the start of the segment, in true distance.
	end_true (label):    Column name of the end of the segment, in true distance.
	split_at (label, list of labels, or bool): Variable(s) other than the `id_vars` by which to split segments where there is discontinuity. If true, all non-reference point labels will be used. If false, splits are only done at discontinuities in `id_vars` and gaps/overlaps in the linear references.	
	summarise (dict or bool):
	Returns:
		(DataFrame) new dataframe with containing all input columns segmented by and summarised on.
	
	Examples
	--------

	>>> df = pd.DataFrame({'ROAD': ['H001', 'H001', 'H001', 'H001', 'H001', 'H001'], 'CWAY': ['S', 'S', 'S', 'S', 'L', 'L'],
	'START_SLK': [12.2, 12.25, 12.3, 12.6, 0, 0.4], 
	'END_SLK': [12.25, 12.3, 12.6, 13, 0.4, 0.86],
	'ROUGHNESS': [0.2, 0.6, 0.7, 0.5, 2, 3],
	'CAT': ['a', 'b', 'b', 'c', 'f', 'f'],
	'SURF_TYPE': ['Asphalt', 'Asphalt', 'Seal', 'Seal', 'Seal', 'Seal']})
	>>> df
	    ROAD  CWAY START_SLK  END_SLK  SURF_TYPE
	0   H001    S   12.2      12.25    Asphalt
	1   H001    S   12.25     12.3     Asphalt
	2   H001    S   12.3      12.6     Seal
	3   H001    S   12.6      13       Seal
	4   H001    L   0         0.4      Seal
	5   H001    L   0.4       0.86     Seal
			
	Get segments for `id_vars` `ROAD` and `CWAY` only.
	>>> get_segments(df, id_vars = ['ROAD', 'CWAY'], start = 'START_SLK', end = 'END_SLK')
	    ROAD  CWAY  START_SLK  END_SLK
	0   H001   L    0.0        0.86
	1   H001   S    12.2       13

	Get segments for `id_vars` `ROAD` and `CWAY` split at changes in `SURF_TYPE`.
	>>> get_segments(df, id_vars = ['ROAD', 'CWAY'], start = 'START_SLK', end = 'END_SLK')
	    ROAD  CWAY  START_SLK  END_SLK SURF_TYPE
	0   H001   L      0.0       0.86    Sealf
	1   H001   S      12.2      12.3    Asphalt
	2   H001   S      12.3      13      Seal

	Split at changes in `SURF_TYPE`, and summarise `ROUGHNESS` and `CAT`, by mean and most common values respectively.
	>>> from pyroads.calc import most_common
	>>> reshape.get_segments(df, id_vars = ['ROAD', 'CWAY'], start = 'START_SLK', end = 'END_SLK', summarise= {'ROUGHNESS': 'mean', 'CAT': most_common}, id = True)	    
	    ROAD  CWAY  START_SLK  END_SLK  SURF_TYPE  ROUGHNESS_mean  CAT_most_common
	0   H001   L     0.0	   0.86      Seal       2.534884         f
	1   H001   S     12.2	   12.3      Seal       0.400000         a
	2   H001   S     12.3	   13        Seal       0.585714         c
	"""
	new_data = data.copy()

	#If using point parameters, make sure they are in metres
	if bool(SLK or true_SLK):
		if not isinstance(segment_size, int):
			raise ValueError("`segment_size` must be provided when using pre-stretched data")
		else:
			if true_SLK is not None:
				if not (new_data[true_SLK].dtype != int or new_data[true_SLK].dtype != np.int64): 
					new_data.loc[:, true_SLK] = as_metres(new_data.loc[:, true_SLK])
			if SLK is not None:
				if not (new_data[SLK].dtype != int or new_data[SLK].dtype != np.int64): 
					new_data.loc[:, SLK] = as_metres(new_data.loc[:, SLK])	
	# Otherwise stretch into equal length segments
	else:
		starts = [col for col in [start, start_true] if col in new_data.columns]
		ends = [col for col in [end, end_true] if col in new_data.columns]
		lengths = as_metres(new_data[ends[0]]) - as_metres(new_data[starts[0]])
		segment_size = int(np.gcd.reduce(lengths))
		new_data = stretch(data, start=start, end=end, start_true=start_true, end_true=end_true, segment_size = segment_size, as_km=False)
		if bool(start):
			SLK = 'SLK'
		if bool(start_true):
			true_SLK = 'true_SLK'
	
	# Detect whether operating on SLK, True SLK, or both.
	SLKs = [slk for slk in [true_SLK, SLK] if slk is not None]
				
	#allow id_vars and split_at to have only one value
	if isinstance(split_at, str):
		split_at = [split_at]
	if isinstance(id_vars, str):
		id_vars = [id_vars]

	#Remove all rows where the SLK and/or id variables are missing
	new_data = new_data.dropna(subset = SLKs + id_vars)

	# split_at - the variables for which to ensure are not broken between segments
	if bool(split_at):
		# If split_at is True, group by all columns other than the `id_vars` (ID variables)
		if split_at is True:
			split_at = [col for col in new_data.columns if col not in id_vars + SLKs]
			if isinstance(summarise, dict):
				split_at = [col for col in new_data.columns if col not in id_vars + SLKs + list(summarise.keys())]
		# Treat all NAs the same
		new_data.loc[:, split_at] = new_data.loc[:, split_at].fillna(-1)
	# If split_at is False, the variable is an empty list
	else:
		split_at = []
	
	# Sort by all id_vars, then SLK, then split_at if declared
	new_data = new_data.drop_duplicates(subset = id_vars + SLKs + split_at)
	new_data = new_data.sort_values(id_vars + SLKs + split_at).reset_index(drop=True)
	
	#Check for Points of Equation
	#If both SLK and true SLK are declared, create a column equal to the difference between the two and check for changes in the variable
	if bool(true_SLK and SLK):
		new_data['slk_diff'] = new_data[true_SLK] - new_data[SLK]
		new_data.insert(0, "groupkey", new_data.groupby(split_at + id_vars + ['slk_diff']).ngroup())
	else:
		new_data.insert(0, "groupkey", new_data.groupby(split_at + id_vars).ngroup())
	
	# Create lag and lead columns for SLK, true SLK, and the split_at key to check whether a new group has started
	for var in SLKs:
		new_data['lead_' + var] = new_data[var].shift(-1, fill_value=0)
		new_data['lag_' + var] = new_data[var].shift(1, fill_value=0)
	new_data['lead_groupkey'] = new_data['groupkey'].shift(-1, fill_value=0)
	new_data['lag_groupkey'] = new_data['groupkey'].shift(1, fill_value=max(new_data['groupkey']))
	# Create columns based on whether they represent the start or end of a section.
	starts, ends = {}, {}
	
	for var in SLKs:
		# if the lagged SLKs are not one observation length less than the actual or the lagged group-key is different.
		starts[var] = np.where(
			(new_data['lag_' + var] == (new_data[var] - segment_size)) & 
			(new_data['lag_groupkey'] == new_data['groupkey']),
			False, True)  
		# if the lead SLKs are not one observation length more than the actual or the lead group-key is different.
		ends[var] = np.where(
			(new_data['lead_' + var] == (new_data[var] + segment_size)) & 
			(new_data['lead_groupkey'] == new_data['groupkey']), 
			False, True)
			
	new_data.loc[:, 'start_bool'] = np.prod([i for i in starts.values()], axis=0, dtype=bool)
	new_data.loc[:, 'end_bool'] = np.prod([i for i in ends.values()], axis=0, dtype=bool)
	
	new_data['segment_id'] = new_data.groupby('groupkey')['start_bool'].cumsum()
	new_data['segment_id'] = new_data.groupby(['groupkey', 'segment_id']).ngroup()
	
	new_data = new_data.drop(['start_bool', 'end_bool', 'groupkey'] + [col for col in new_data.columns if 'lag' in col or 'lead' in col], axis=1)
	
	#Summarise the data by `segment_id`
	#By default, summarise the SLK into min and max columns, representing Start and End SLK respectively
	if bool(summarise):
		agg_dict = {SLK: [min, max] for SLK in SLKs}
		# If an aggregation dictionary is provided to `summarise`, add the methods to the SLK method detailed in the previous step
		if isinstance(summarise, dict):
			agg_dict.update(summarise)
	new_data = new_data.groupby(['segment_id'] + id_vars + split_at).agg(agg_dict)
	new_data.columns = ["_".join(x) for x in new_data.columns]
	new_data = new_data.rename(columns={'SLK_min': 'START_SLK', 'SLK_max': 'END_SLK', 'true_SLK_min': 'START_TRUE', 'true_SLK_max': 'END_TRUE'})  # pyright: ignore[reportCallIssue]
	
	start_cols = [col for col in ['START_SLK', 'START_TRUE'] if col in new_data.columns]
	end_cols = [col for col in ['END_SLK', 'END_TRUE'] if col in new_data.columns]
	slks = start_cols + end_cols
	
	# Increment slk_ends by the observation length
	for col in end_cols:
		new_data[col] = new_data[col] + segment_size
	
	# Turn into km
	new_data.loc[:, slks] = new_data[slks]/1000
	
	if bool(summarise):
		# Add the groupbys back to the columns
		new_data = new_data.reset_index('segment_id', drop=True)
		new_data = new_data.reset_index()
	else:
		new_data['segment_id'] = [i for i in range(len(new_data))]
	
	# After the aggregations are done, the missing data can go back to being NaN
	new_data.loc[:, split_at] = new_data.loc[:, split_at].replace(-1, np.nan)
	new_data = new_data.sort_values(id_vars + start_cols + end_cols)

	#sort columns
	new_data = new_data[id_vars + slks + split_at + [col for col in new_data.columns if col not in id_vars + slks + split_at]]

	new_data = new_data.reset_index(drop=True)
	
	return new_data