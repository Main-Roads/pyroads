name_to_id = {
	'Asphalt': 10,
	'Clay': 5,
	'Concrete': 6,
	'Crushed Rock': 3,
	'Ferricrete': 11,
	'Gravel': 2,
	'Limestone': 7,
	'HCT': 9,
	'Recycled Material': 8,
	'Sand': 4,
	'Sand Clay': 1,
	'Unknown': 0
}
"""
Map from pavement type name to pavement id

Examples:
	>>> df["pavement_type_id"] = df["pavement_type_name"].map(map_pavement_type_name_to_id)
"""


id_to_name = {
	10: 'Asphalt',
	5: 'Clay',
	6: 'Concrete',
	3: 'Crushed Rock',
	11: 'Ferricrete',
	2: 'Gravel',
	7: 'Limestone',
	9: 'HCT',
	8: 'Recycled Material',
	4: 'Sand',
	1: 'Sand Clay',
	0: 'Unknown',
}

map_pavement_type_name_to_id = name_to_id
map_pavement_type_id_to_name = id_to_name
"""
Map from pavement type name to pavement id

Examples:
	>>> df["pavement_type_name"] = df["pavement_type_id"].map(id_to_name)
"""