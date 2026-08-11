


import pytest
from contextlib import nullcontext as does_not_raise


@pytest.mark.parametrize(
	"test_input,expected_result,expectation",
	[
		(
			{"measure_from":190,  "measure_to":195,   "multiples":50},
			[190, 195], 
			does_not_raise()
		),
		(
			{"measure_from":190,  "measure_to":270,   "multiples":50},
			[190, 200, 250, 270], 
			does_not_raise()
		),
		(
			{"measure_from":190,  "measure_to":270,   "multiples":45},
			[190, 225, 270], 
			does_not_raise()
		),
		(
			{"measure_from":190,  "measure_to":225,   "multiples":10},
			[190, 200, 210, 220, 225], 
			does_not_raise()
		),
		(
			{"measure_from":14.2, "measure_to":16.3,  "multiples":0.3},
			[14.2, 14.4, 14.7, 15.0, 15.3, 15.6, 15.9, 16.2, 16.3], 
			does_not_raise()
		),
		(
			{"measure_from":14.2, "measure_to":16.05, "multiples":0.5, "minimum_length":0.1},
		 	[14.2, 14.5, 15.0, 15.5, 16.05],
			 does_not_raise()
		),
		(
			{"measure_from":13.95, "measure_to":15.6, "multiples":0.5, "minimum_length":0.1},
		 	[13.95, 14.5, 15.0, 15.5, 15.6],
			 does_not_raise()
		),
		(
			{"measure_from":14.05, "measure_to":15.95, "multiples":0.5, "minimum_length":0.1},
		 	[14.05, 14.5, 15.0, 15.5, 15.95],
			 does_not_raise()
		),
		(
			{"measure_from":14.05, "measure_to":14.05, "multiples":0.5, "minimum_length":0.1},
		 	[14.05, 14.05],
			 does_not_raise()
		),
		(
			{"measure_from":14.05, "measure_to":14.00, "multiples":0.5, "minimum_length":0.1},
			None,
			pytest.raises(ValueError)
		),
	]
)


def test_linspace_steps(test_input, expected_result, expectation):
	import numpy as np
	from pyroads.segmenter import linspace_steps

	with expectation:
		actual_result = linspace_steps(**test_input)

		expected_result = np.array(expected_result)

		assert len(actual_result) == len(expected_result)
		assert np.isclose(actual_result, expected_result).all()
