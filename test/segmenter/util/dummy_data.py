import pandas as pd

var_a:pd.Series = pd.Series([
	1,
	2,
	3,
	4,
	2,
	3,
	1,
	5,
	3,
	4,
	6,
	4,
	3,
	7,
	5,
	6,
	4,
	5
])

var_b:pd.Series = pd.Series([
	3,
	2,
	1,
	3,
	5,
	6,
	5,
	6,
	4,
	7,
	9,
	4,
	3,
	2,
	3,
	1,
	3,
	2
])

slk_from:pd.Series = pd.Series([
	0.000,
	0.010,
	0.020,
	0.030,
	0.040,
	0.050,
	0.060,
	0.070,
	0.080,
	0.090,
	0.100,
	0.110,
	0.120,
	0.130,
	0.140,
	0.150,
	0.160,
	0.170
])

slk_to:pd.Series = pd.Series([
	0.010,
	0.020,
	0.030,
	0.040,
	0.050,
	0.060,
	0.070,
	0.080,
	0.090,
	0.100,
	0.110,
	0.120,
	0.130,
	0.140,
	0.150,
	0.160,
	0.170,
	0.180
])

length:pd.Series = slk_to - slk_from



df1 = pd.concat([var_a, var_b, slk_from, slk_to, length], axis=1,keys=["var_a", "var_b", "slk_from", "slk_to", "length"])