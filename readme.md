# pyroads

[![Run Tests](https://github.com/Main-Roads/pyroads/actions/workflows/run_tests.yml/badge.svg)](https://github.com/Main-Roads/pyroads/actions/workflows/run_tests.yml)

Python package for analysis of road asset data

## Installation

### If you have git installed
```
pip install git+https://github.com/shaan-nmb/pyroads.git#egg=pyroads
```

### If you do not have git installed
```
pip install "https://github.com/shaan-nmb/pyroads/zipball/main/"
```
## Background

This package collects together a number of functions used to reshape and prepare road condition data for various modelling and reporting processes.

## Usage

```python
from pyroads.reshape import stretch, interval_merge, make_segments, get_segments

from pyroads.mappings.cross_sectional_position import cway_to_side, hsd_to_side, lane_to_side, get_lanes
from pyroads.mappings.link_category import map_link_category_to_mabcd
from pyroads.mappings.pavement import map_pavement_type_name_to_id, map_pavement_type_id_to_name
from pyroads.mappings.region import map_region_name_to_number, map_region_number_to_region_name
from pyroads.mappings.route import route_change, route_description
from pyroads.mappings.rurality import rurality
from pyroads.mappings.surface import surf_id, surf_type

from pyroads.misc.custom_aggregations import first, last,most,q75,q90,q95
from pyroads.misc.deteriorate import deteriorate
```

## Future Improvements

- [x] Directory structure to be reshuffled to allow more ergonomic imports
- [ ] The `widen_by_lane()` function is not working / is deprecated and needs to be rewritten.
- [ ] The `map_link_category_to_mabcd()` is deprecated / needs simplification
- [ ] `mappings` could perhaps be separated into a new package
- [ ] `gcd_list` is deprecated and needs to be removed in a future update. `numpy.gcd.reduce` is the same
- [ ] some mappings are functions, and some mappings are just dicts. Probably just dicts are good? See _region.py_
- [ ] `rurality()` is deprecated as it will probably change over time and should be part of another library

