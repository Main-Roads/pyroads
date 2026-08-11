from typing import Literal

import pandas as pd

id_to_name = {
	1: "Asphalt Dense Graded",
	2: "Asphalt Intersection Mix",
	3: "Asphalt Open Graded",
	4: "Concrete",
	5: "Paving",
	6: "Primer Seal",
	7: "Rubberised Seal",
	8: "Slurry Seal",
	9: "Single Seal",
	10: "Two Coat Seal",
	11: "Asphalt Stone Mastic",
	12: "Asphalt Open Graded on Dense Graded",
	13: "Asphalt Gap Graded Rubberised (GGAR)"
}

name_to_id =  dict((name, id) for id,name in id_to_name.items())


def surf_type(data: pd.DataFrame, source: str = "surface_type", to: Literal["full", "short", "asphalt", "grouped", "group_id"] = "short", **kwargs) -> pd.Series:
	source = kwargs.pop("from", source)
	surf_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
	full = ["Asphalt Dense Graded", "Asphalt Intersection Mix", "Asphalt Open Graded", "Concrete", "Paving", "Primer Seal", "Rubberised Seal", "Single Seal", "Slurry Seal", "Two Coat Seal", "Asphalt Stone Mastic", "Asphalt Open Graded on Dense Graded", "Asphalt Gap Graded Rubberised (GGAR)"]
	short = ["DGA", "IMA", "OGA", "Concrete", "Paving", "Primer Seal", "Rubberised Seal", "Single Seal", "Slurry Seal", "Two Coat Seal", "SMA", "OGA on DGA", "GGAR"]
	asphalt = ["DGA", "IMA", "OGA", "Other", "Other", "Seal", "Seal", "Seal", "Seal", "Seal", "SMA", "OGA on DGA", "GGAR"]
	grouped = ["Asphalt", "Asphalt", "Asphalt", "Other", "Other", "Seal", "Seal", "Seal", "Seal", "Seal", "Asphalt", "Asphalt", "Asphalt", 'Asphalt']
	group_id = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 1, 1, 1, 1]
	
	if to == "full":
		surf_dict = dict(zip(surf_id, full))
	elif to == "short":
		surf_dict = dict(zip(surf_id, short))
	elif to == "asphalt":
		surf_dict = dict(zip(surf_id, asphalt))
	elif to == "grouped":
		surf_dict = dict(zip(surf_id, grouped))
	elif to == 'group_id':
		surf_dict = dict(zip(surf_id, group_id))
	else:
		raise ValueError("ERROR surf_type(to): Please choose a to from ['full', 'short', 'asphalt', grouped'].")
	
	return data[source].map(surf_dict)


def surf_id(data: pd.DataFrame, source: str = "surface_type", to="short", **kwargs) -> pd.Series:
	source = kwargs.pop("from", source)
	"""
	convert surface type id to text
	"""
	
	surf_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
	full = ["Asphalt Dense Graded", "Asphalt Intersection Mix", "Asphalt Open Graded", "Concrete", "Paving", "Primer Seal", "Rubberised Seal", "Single Seal", "Slurry Seal", "Two Coat Seal", "Asphalt Stone Mastic", "Asphalt Open Graded on Dense Graded", 'Asphalt Gap Graded Rubberised (GGAR)']
	short = ["DGA", "IMA", "OGA", "Concrete", "Paving", "Primer Seal", "Rubberised Seal", "Single Seal", "Slurry Seal", "Two Coat Seal", "SMA", "OGA on DGA"]
	asphalt = ["DGA", "IMA", "OGA", "Other", "Other", "Seal", "Seal", "Seal", "Seal", "Seal", "SMA", "OGA on DGA", 'GGAR']
	grouped = ["Asphalt", "Asphalt", "Asphalt", "Other", "Other", "Seal", "Seal", "Seal", "Seal", "Seal", "Asphalt", "Asphalt", "Asphalt", 'Asphalt']
	group_id = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 1, 1, 1, 1]
	
	if to == "full":
		surf_dict = dict(zip(full, surf_id))
	elif to == "short":
		surf_dict = dict(zip(short, surf_id))
	elif to == "asphalt":
		surf_dict = dict(zip(asphalt, surf_id))
	elif to == "grouped":
		surf_dict = dict(zip(grouped, surf_id))
	elif to == 'group_id':
		surf_dict = dict(zip(group_id, surf_id))
	else:
		raise ValueError("ERROR surf_type(to): Please choose a to from ['full', 'short', 'asphalt', grouped'].")
	
	return data[source].map(surf_dict)

