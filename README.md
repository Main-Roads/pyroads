# pyroads

`pyroads` is a Python toolkit for preparing, segmenting, merging, and analysing
linear road-asset data. It consolidates the former `segmenter`,
`homogeneous-segmentation`, and `merge-segments` projects behind one package.

## Installation

```bash
pip install pyroads
```

For development from a checkout:

```bash
uv sync
uv run pytest
```

Numba is a required dependency because the default interval merge and several
segmentation kernels use compiled NumPy/Numba implementations.

## Public API

### Reshaping and mappings

```python
from pyroads.reshape import get_segments, interval_merge, make_segments, stretch
from pyroads.mappings.cross_sectional_position import cway_to_side, get_lanes
from pyroads.mappings.pavement import map_pavement_type_name_to_id
from pyroads.mappings.route import route_change, route_description
from pyroads.mappings.surface import surf_id, surf_type
from pyroads.calc import first, last, most_common, q75, q90, q95
```

### Segmenting road data

```python
from pyroads.segmenter import (
	check_linear_index,
	cross_sections_normalised,
	fetch_road_network_info,
	segment_by_categories_and_slk_discontinuities,
	segment_by_categories_and_slk_true_discontinuities,
	split_rows_by_category_to_max_segment_length,
	split_rows_by_segmentation,
)
```

These functions identify category and linear-index discontinuities, split rows
into regular lengths, validate segmentation indexes, and calculate normalized
cross-sections.

### Homogeneous segmentation

```python
from pyroads.homogeneous_segmentation import (
	segment_ids_to_maximize_spatial_heterogeneity,
	segment_ids_to_minimize_coefficient_of_variation,
)
```

Both methods accept a DataFrame, a pair of linear-measure columns, condition
columns, and an allowed segment-length range.

### Merging interval data

```python
from pyroads.merge import Action, Aggregation, on_slk_intervals

result = on_slk_intervals(
	target=segmentation,
	data=measurements,
	join_left=["road", "cwy"],
	column_actions=[
		Action("roughness", Aggregation.LengthWeightedAverage()),
		Action("surface", Aggregation.KeepLongest()),
	],
	from_to=("slk_from", "slk_to"),
)
```

`on_slk_intervals()` uses the optimized implementation by default. The
compatibility flag remains available: pass `legacy=True` to use the legacy
implementation, or `legacy=False` explicitly to select the optimized path.
Optional `polars` and `dask` extras provide alternate backends.

### Fetching Main Roads WA data

`fetch_road_network_info()` returns a Pandas DataFrame. It does not write a
CSV or Parquet file automatically; the complete response is assembled in
memory. The default request selects road attributes without geometry and uses
stable `OBJECTID` pagination.

```python
roads = fetch_road_network_info(
	additional_params={
		"where": "ROAD = 'H001'",
		"outFields": "ROAD,START_SLK,END_SLK,OBJECTID",
	},
	chunk_limit=1,
)

roads.to_parquet("road_network.parquet", index=False)
```

Use `returnGeometry=True` when geometry is required, or pass `url` to a
compatible ArcGIS REST endpoint. `query_params` replaces all defaults;
`additional_params` and legacy keyword arguments override selected defaults.

## Compatibility

The consolidated package preserves the public function behavior while moving
the import paths under `pyroads`. Existing standalone repositories remain
useful as historical references, but new code should import from:

- `pyroads.segmenter`
- `pyroads.homogeneous_segmentation`
- `pyroads.merge`

## Examples and tests

Runnable example scripts are under [`examples/`](examples/), including merge
benchmarks and chart generation. Manual notebooks are under
[`examples/segmenter/`](examples/segmenter/). Documentation images from the
former projects are under [`readme_extras/`](readme_extras/).

The consolidated automated tests are under [`test/`](test/) and are grouped by
functionality. Run them with:

```bash
uv run pytest
```

## Acknowledgements

The original `pyroads` package was created by Shaan Ciantar. Nicholas Archer
wrote most of the functionality in the former `merge-segments`,
`homogeneous-segmentation`, and `segmenter` packages.

Dagmawi Tadesse consolidated these packages into the `pyroads` repository,
preserving their public APIs for backward compatibility and delivering
significant performance improvements through NumPy and Numba JIT compilation.

## Disclaimer

This software is made available as an open-source project by Main Roads
Western Australia.

The software is provided for general use and research purposes. Users are
responsible for validating the suitability of the software for their own
applications.

Main Roads Western Australia does not accept responsibility for decisions,
analyses, asset management outcomes, or other results produced using this
software.