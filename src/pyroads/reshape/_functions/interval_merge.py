from typing import Any

from .get_segments import get_segments
from .stretch import stretch
from .as_metres import as_metres
import numpy as np


def interval_merge(
	left: Any,
	right: Any,
	id_vars: Any = None,
	start: Any = None,
	end: Any = None,
	start_true: Any = None,
	end_true: Any = None,
	left_id_vars: Any = None,
	right_id_vars: Any = None,
	start_left: Any = None,
	end_left: Any = None,
	start_right: Any = None,
	end_right: Any = None,
	start_true_left: Any = None,
	end_true_left: Any = None,
	start_true_right: Any = None,
	end_true_right: Any = None,
	split_at: Any = 'left',
	summarise: Any = True,
	use_ranges: Any = True):
	
	#Set all of the right and left variables depending on what parameters the user chose
	if id_vars is not None:
		if left_id_vars is None:
			left_id_vars = id_vars
		else:
			if isinstance(left_id_vars, str) and len(left_id_vars.split()) == 1:
				left_id_vars = [left_id_vars]
			left_id_vars = id_vars + left_id_vars
		if right_id_vars is None:
			right_id_vars = id_vars
		else:
			if isinstance(right_id_vars, str) and len(right_id_vars.split()) == 1:
				right_id_vars = [right_id_vars]
			right_id_vars = id_vars + right_id_vars
	if start is not None:
		start_left, start_right = start, start
	if end is not None:
		end_left, end_right = end, end
	if start_true is not None:
		start_true_left, start_true_right = start_true, start_true
	if end_true is not None:
		end_true_left, end_true_right = end_true, end_true

	# Define the interval columns for the datasets
	starts_left = [start for start in [start_left, start_true_left] if start is not None]
	ends_left = [end for end in [end_left, end_true_left] if end is not None]
	starts_right = [start for start in [start_right, start_true_right] if start is not None]
	ends_right = [end for end in [end_right, end_true_right] if end is not None]
	slk_interval_cols = starts_left + starts_right + ends_left + ends_right
	
	# Create copies as to not change the original data
	left_copy = left.copy()
	right_copy = right.copy()
	#drop segment id columns if in datasets
	if 'segment_id' in left.columns:
		left_copy = left_copy.drop('segment_id', axis=1)
	if 'segment_id' in right_copy.columns:
		right_copy = right_copy.drop('segment_id', axis=1)
	

	# Get a list of all the columns that will be used, and drop all of the rest

	## List of columns that will be summarised on
	if isinstance(summarise, dict):
		summarise_cols = list(summarise.keys())
	else:
		summarise_cols = []		
	## List of ID variables
	### Find the number of ID variables to join on
	id_len = min(len(left_id_vars), len(right_id_vars))  # max number of mutual IDs
	### Rename the right ID Vars to be congruent with the left
	id_dict = dict(zip(right_id_vars[0:id_len], left_id_vars[0:id_len]))
	right_copy = right_copy.rename(columns=id_dict)
	id_vars = left_id_vars[0:id_len]
	## List of columns that will be grouped on
	if isinstance(split_at, list):
		split_at = split_at
	elif split_at is True:
		split_at = [col for col in [col for col in left_copy.columns]  + [col for col in right_copy.columns] if col not in slk_interval_cols + id_vars + summarise_cols] + [var for var in left_id_vars + right_id_vars if var not in id_vars]
	elif split_at == 'left':
		split_at = [col for col in left_copy.columns if col not in starts_left + ends_left + left_id_vars + summarise_cols] + [var for var in left_id_vars if var not in id_vars]
	elif split_at == 'right':
		split_at = [col for col in right_copy.columns if col not in starts_right + ends_right + id_vars + summarise_cols] + [var for var in left_id_vars if var not in id_vars]
	else:
		split_at = []
	
	## Add the lists of id_vars, summarise_cols, slk_intervals, and split_at together as the columns to keep
	keep_cols = id_vars+ slk_interval_cols + split_at + summarise_cols

	## Drop everything outside of the keep cols
	right_copy = right_copy.loc[:, [col for col in right_copy.columns if col in keep_cols]]
	left_copy = left_copy.loc[:, [col for col in left_copy.columns if col in keep_cols]]
	
	# Stretch both dataframes by a shared Greatest Common Divisor (GCD) of lengths

	##convert SLKs to integer metres for easier operations
	left_metres = left_copy.loc[:, starts_left + ends_left].apply(as_metres)
	right_metres = right_copy.loc[:, starts_right + ends_right].apply(as_metres)

	## Find the GCD
	### Create a list of all the lengths	
	lengths: tuple[Any, ...] = ()
	# left
	for start, end in zip(starts_left, ends_left):
		lengths = tuple(left_metres[end] - left_metres[start])
	# right
	for start, end in zip(starts_right, ends_right):
		lengths = lengths + tuple(right_metres[end] - right_metres[start])
	# Find the minimum
	gcd = np.gcd.reduce(lengths)
	
	# rename SLKs congruently
	#Dictionary with which to rename the SLK columns
	slk_dict: dict[Any, Any] = {
		start_right: 'START_SLK' if bool(start_right) else None, 
		start_true_right: 'START_TRUE' if bool(start_true_right) else None, 
		end_right: 'END_SLK' if bool(end_right) else None, 
		end_true_right: 'END_TRUE' if bool(end_true_right) else None, 
		start_left: 'START_SLK' if bool(start_left) else None, 
		start_true_left: 'START_TRUE' if bool(start_true_left) else None, 
		end_left: 'END_SLK' if bool(end_left) else None, 
		end_true_left: 'END_TRUE' if bool(end_true_left) else None}
	##Don't include the missing parameters
	for key in list(slk_dict.keys()):
		if key is None:
			slk_dict[key] = None
	##rename
	left_copy = left_copy.rename(columns=slk_dict)
	right_copy = right_copy.rename(columns=slk_dict)
	
	#Stretch
	left_stretched = stretch(left_copy, start=slk_dict[start_left], end=slk_dict[end_left], start_true=slk_dict[start_true_left], end_true=slk_dict[end_true_left], segment_size = gcd, keep_ranges=use_ranges)	
	right_stretched = stretch(right_copy, start=slk_dict[start_right], end=slk_dict[end_right], start_true=slk_dict[start_true_right], end_true=slk_dict[end_true_right], segment_size = gcd)
	
	# Now that they have the same names, the mutual ID variables are the same as the parameter left_id_vars to the maximum mutual length of the id_vars
	left_stretched = left_stretched.set_index(id_vars + [col for col in ['SLK', 'true_SLK'] if col in left_stretched.columns])
	right_stretched = right_stretched.set_index(id_vars + [col for col in ['SLK', 'true_SLK'] if col in right_stretched.columns])
	
	# join by index
	joined = left_stretched.join(right_stretched, how='left')

	if use_ranges:
		# change the name of the original SLKs before creating segments to avoid confusion
		slks = list(set([i for i in list(slk_dict.values()) if i in joined.columns]))
		joined.columns = ['org_' + col if col in slks else col for col in joined.columns]
		org_slks = ['org_' + i for i in slks]
		id_vars = id_vars + org_slks
		joined = joined.reset_index()
	else:
		slks = []
		joined = joined[~joined.index.duplicated(keep='first')].reset_index()

	summarised_df = get_segments(joined, true_SLK='true_SLK' if 'true_SLK' in joined.columns else None, SLK='SLK' if 'SLK' in joined.columns else None, segment_size = int(gcd), id_vars=id_vars, split_at=split_at, summarise=summarise)
	
	# Drop the duplicates of the SLK columns caused by `get_segments` if the original ranges are being used for the segments
	if use_ranges:
		summarised_df = summarised_df.drop(slks, axis=1)
		summarised_df.columns = [col[4:] if col[:4] == "org_" else col for col in summarised_df.columns]
	
	#Order the columns to be the id_vars followed by the SLKs
	for slk in slks:
		x = 0
		summarised_df.insert(len(id_vars) + x, slk, summarised_df.pop(slk))
		x = x + 1
	
	summarised_df = summarised_df.drop_duplicates()
	summarised_df = summarised_df.reset_index(drop=True)
	
	return summarised_df
