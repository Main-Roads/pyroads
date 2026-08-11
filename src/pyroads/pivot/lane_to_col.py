from typing import Any, Optional, Union
import numpy as np

from .lane_side_split import lane_side_split as split_lanes_by_side



def lane_to_col(
	data: Any,
	xsp: str,
	side_split: bool = False,
	id_vars: Optional[list[str]] = None,
	values: Optional[Union[str, list[str]]] = None,
	turn_pockets: bool = False,
	all_lanes: bool = False,
	):
	
	#Make a copy of the data frame
	new_data = data.copy()

	if not all_lanes:
		#Select only regular lanes
		regular_lanes = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'R1', 'R2', 'R3', 'R4', 'R5']
		if turn_pockets:
			regular_lanes.append('TP')		
		new_data = new_data.loc[new_data[xsp].isin(regular_lanes)]
	else: 
		lanes = ['L1', 'L2', 'L3', 'L4', 'LL1', 'R1', 'R2', 'LR1', 'L', 'RR1', 'R', 'R3', 'LR2', 'LR3', 'RL1', 'RO', 'LO', 'RR2', 'RL2', 'ER', 'R4', 'R5', 'L5', 'L6', 'LL2', 'RR3', 'LL3']
		new_data = new_data.loc[new_data[xsp].isin(lanes)]
	
	if isinstance(values, str):
		values = [values]
	if values is None:
		values = []
	
	if id_vars is None:
		id_vars = [col for col in new_data.columns if col not in values + ['LANE_NO', xsp]]
		
	
	
	if side_split:
		new_data = split_lanes_by_side(new_data, xsp = xsp)
	#pivot
		new_data = new_data.fillna(-1).pivot_table(index = id_vars, columns = 'LANE_NO', values = values, aggfunc = 'first').reset_index()
	else:
		new_data = new_data.fillna(-1).pivot_table(index = id_vars, columns = xsp, values = values, aggfunc = 'first').reset_index()
	
	new_data = new_data.replace(-1, np.nan)
	new_data.columns = ["".join(x) for x in new_data.columns]
	
	return new_data