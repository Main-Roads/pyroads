# from typing import Literal



# def ra_transform(x: pd.Series, method: Literal["name", "number"] = "name") -> pd.Series:
# 	number = [1, 2, 5, 6, 7, 8, 11, 14]
# 	name = ["Great Southern", "South West", "Goldfields-Esperance", "Kimberley", "Metropolitan", "Wheatbelt", "Pilbara", "Mid West-Gascoyne"]
#
# 	if method == "name":
# 		ra_dict = dict(zip(name, number))
# 	if method == "number":
# 		ra_dict = dict(zip(number, name))
# 	else:
# 		raise ValueError()
#
# 	ra = x.map(ra_dict).astype(pd.Int64Dtype())
#
# 	return ra


name_to_number = {
	"Great Southern":         1,
	"South West":             2,
	"Goldfields - Esperance": 5,
	"Kimberley":              6,
	"Metropolitan":           7,
	"Wheatbelt":              8,
	"Pilbara":                11,
	"Mid West-Gascoyne":      14,
}
"""
Map from region name string to region number

Examples:
	>>> df["region_id"] = df["region_name"].map(name_to_number)
	>>> df["region_id"] = df["region_name"].apply(name_to_number.get, args=("Some Default Value"))
	
"""

number_to_name = {
	1:  "Great Southern",
	2:  "South West",
	5:  "Goldfields - Esperance",
	6:  "Kimberley",
	7:  "Metropolitan",
	8:  "Wheatbelt",
	11: "Pilbara",
	14: "Mid West-Gascoyne",
}

map_region_name_to_number = name_to_number
map_region_number_to_region_name = number_to_name
"""
Map from region number to region name

Examples:
	>>> df["region_name"] = df["region_id"].map(number_name)
	>>> df["region_name"] = df["region_id"].apply(number_to_name.get, args=("Some Default Value"))
"""