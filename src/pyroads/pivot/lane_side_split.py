
from xmlrpc.client import Boolean
from typing import Any

def lane_side_split(
	data: Any,
	xsp: str,
	all_lanes: Boolean = False,
):
	new_data = data.copy()

	if not all_lanes:
		new_data = new_data[new_data[xsp].str.contains("^((L|R)[1-6])", regex = True)]
	
	#Split into two columns. One for direction, and one for lane number.
	new_data['SIDE'] = new_data[xsp].str[0]
	new_data['LANE_NO'] = new_data[xsp].str[1:].astype(int)
	#Drop the XSP column and add 'DIRN' to the id_vars
	new_data = new_data.drop(xsp, axis = 1)
	return new_data