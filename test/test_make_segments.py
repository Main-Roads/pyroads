
def test_make_segments_slk_only():
	import pandas as pd
	from pyroads.reshape import make_segments
	
	data_to_segment = pd.DataFrame(
		columns=["road_no", "carriageway", "slk_from", "slk_to", "value"],
		data=[
			["H001", "L", 0.010, 0.050, "a"],
			["H001", "L", 0.050, 0.080, "b"],
			["H001", "L", 0.080, 0.100, "c"],
			["H001", "L", 0.100, 0.120, "d"],
		]
	)
	
	actual_result = make_segments(
		data_to_segment,
		start="slk_from",
		end="slk_to",
		max_segment=10,
		as_km=True
	)
	
	expected_result = pd.DataFrame(
		columns=["road_no", "carriageway", "slk_from", "slk_to", "Length", "value"],
		data=[
			["H001", "L", 0.010, 0.020, 0.010, "a"],
			["H001", "L", 0.020, 0.030, 0.010, "a"],
			["H001", "L", 0.030, 0.040, 0.010, "a"],
			["H001", "L", 0.040, 0.050, 0.010, "a"],
			["H001", "L", 0.050, 0.060, 0.010, "b"],
			["H001", "L", 0.060, 0.070, 0.010, "b"],
			["H001", "L", 0.070, 0.080, 0.010, "b"],
			["H001", "L", 0.080, 0.090, 0.010, "c"],
			["H001", "L", 0.090, 0.100, 0.010, "c"],
			["H001", "L", 0.100, 0.110, 0.010, "d"],
			["H001", "L", 0.110, 0.120, 0.010, "d"],
		]
	)
	# length column might be slightly fuzzy
	actual_result["Length"] = actual_result["Length"].round(3)
	

	pd.testing.assert_frame_equal(
		actual_result,
		expected_result,
		check_like=True, # ignore column / row order
	)
