from typing import Literal

import pandas as pd


def get_lanes(
	x, 
	current: Literal['xsp', 'description', 'side', 'dtims', 'lane'], 
	new: Literal['xsp', 'description', 'side', 'dtims', 'lane']
	):
	
	xsp = pd.Series(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'R1', 'R2', 'R3', 'R4', 'R5', 'LL1', 'LL2', 'LL3', 'LR1', 'LR2', 'LR3', 'RL1', 'RL2', 'RL3', 'RR1', 'RR2', 'RR3', 'LO', 'RO', 'L', 'R'])
	lane = xsp
	description = ["Left lane 2", "Left lane 3", "Left lane 4", "Left lane 5", "Left lane 6", "Right lane 1", "Right lane 2", "Right lane 3", "Right lane 4", "Right lane 5", "Left carriageway, left turn pocket 1", "Left carriageway, left turn pocket 2", "Left carriageway, left turn pocket 3", "Left carriageway, right turn pocket 1", "Left carriageway, right turn pocket 2", "Left carriageway, right turn pocket 3", "Right carriageway, left turn pocket 1", "Right carriageway, left turn pocket 2", "Right carriageway, left turn pocket 3", "Right carriageway, right turn pocket 1", "Right carriageway, right turn pocket 2", "Right carriageway, right turn pocket 3", "Left overtaking lane", "Right overtaking lane", "Left shoulder", "Right shoulder"]
	dtims = [1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 21, 23, 25, 31, 33, 35, 22, 24, 26, 32, 34, 36, 41, 42, 51, 52]
	
	if current == 'xsp':
		old_data = xsp
	elif current == 'description':
		old_data = description
	elif current == 'lane':
		old_data = lane
	elif current == 'dtims':
		old_data = dtims
	else:
		raise ValueError("current must be one of 'xsp' 'description' 'lane' or 'dtims'")
	
	if new == 'xsp':
		new_data = xsp
	elif new == 'description':
		new_data = description
	elif new == 'lane':
		new_data = lane
	elif new == 'dtims':
		new_data = dtims
	else:
		raise ValueError("new must be one of 'xsp' 'description' 'lane' or 'dtims'")
	
	lane_dict = dict(zip(old_data, new_data))
	
	return x.map(lane_dict)