# pyroads

`pyroads` is a Python toolkit for preparing, segmenting, merging, and analysing
linear road-asset data. It consolidates the former `segmenter`,
`homogeneous-segmentation`, and `merge-segments` projects behind one package.

## Installation

```bash
pip install git+https://github.com/Main-Roads/pyroads.git@release
```

Installing directly from GitHub builds the Rust extension locally, so the
machine must have a Rust toolchain with `cargo` and `rustc` available. The build
frontend installs Maturin automatically. If the Rust extension cannot be
imported at runtime, pyroads uses its Numba/Python fallback implementations.

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

## Developer guide

### Development prerequisites

Development requires:

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- A stable Rust toolchain providing `cargo` and `rustc`
- A C build environment supported by PyO3 on the target operating system

Install Rust with [rustup](https://rustup.rs/) when it is not already managed
by the development machine. Check the tools before building:

```bash
python --version
uv --version
rustc --version
cargo --version
```

### Create the development environment

From the repository root, synchronize the locked Python environment and install
the development dependencies:

```bash
uv sync --dev
```

The root `pyproject.toml` uses Maturin as its build backend. The Rust manifest
is `rust/_native/Cargo.toml`, and the compiled extension is packaged as
`pyroads._native`.

### Run tests and quality checks

Run the complete Python test suite:

```bash
uv run pytest
```

Run a focused test file or test function while iterating:

```bash
uv run pytest test/test_interval_merge.py
uv run pytest test/test_interval_merge.py::test_interval_merge
```

Run the configured static checks:

```bash
uv run ruff check src
uv run pyright src
```

The Rust crate can be checked independently:

```bash
cargo fmt --manifest-path rust/_native/Cargo.toml -- --check
cargo check --manifest-path rust/_native/Cargo.toml
```

### Build and inspect distribution artifacts

Build the wheel and source distribution into a temporary directory:

```bash
rm -rf /tmp/pyroads-dist
uv build --out-dir /tmp/pyroads-dist
```

The wheel should contain a platform-specific file named like
`pyroads/_native.cpython-311-x86_64-linux-gnu.so` (the suffix varies by
operating system and Python version). The source distribution should contain
`rust/_native/Cargo.toml` and `rust/_native/src/lib.rs`.

Inspect the artifacts from the command line:

```bash
find /tmp/pyroads-dist -maxdepth 1 -type f -printf '%f\n'
unzip -Z1 /tmp/pyroads-dist/pyroads-*.whl | grep -E 'pyroads/_native|\.so$|\.pyd$'
tar -tzf /tmp/pyroads-dist/pyroads-*.tar.gz | grep 'rust/_native/'
```

Test a built wheel in an isolated environment before distributing it:

```bash
rm -rf /tmp/pyroads-wheel-venv
uv venv /tmp/pyroads-wheel-venv
uv pip install --python /tmp/pyroads-wheel-venv/bin/python /tmp/pyroads-dist/pyroads-*.whl
/tmp/pyroads-wheel-venv/bin/python -c \
	"import numpy as np; from pyroads import _native; print(_native.cumulative_p(np.array([1., 2., 3.])))"
```

On Windows, replace the virtual environment Python path with
`/tmp/pyroads-wheel-venv/Scripts/python.exe` or the equivalent local path.

### Native backend behavior

The Python modules import `pyroads._native` when available. If the extension
cannot be imported, the existing Numba/Python implementations remain usable
and print a fallback notice. No environment variable is needed to select the
backend.

For local Rust work, build the extension in release mode through the root
package configuration:

```bash
uv build --out-dir /tmp/pyroads-dist
```

For a quick Rust-only compile check, use Cargo instead of producing a wheel:

```bash
cargo check --manifest-path rust/_native/Cargo.toml
```

Do not commit locally generated `.so`, `.pyd`, `target/`, virtual-environment,
or cache files. They are ignored by the repository. Native binaries are
platform- and Python-version-specific; distributable wheels must be built for
each supported platform and interpreter combination.

### Command-line interface and benchmarks

The package currently has no dedicated `pyroads` console command. The
supported command-line developer tools are the repository scripts and standard
Python tooling.

Run the segmenter benchmark:

```bash
uv run python examples/segmenter/benchmark.py --repeats 7
```

It prints JSON timing and output-row data for category segmentation, overlay,
cross-sections, and maximum-length splitting.

Run the merge benchmark with synthetic data:

```bash
uv run python examples/merge/compare_merges.py \
	--targets 5000 \
	--data 15000 \
	--groups 5 \
	--repeats 5
```

The merge benchmark can also compare CSV inputs:

```bash
uv run python examples/merge/compare_merges.py \
	--target-file target.csv \
	--data-file data.csv \
	--repeats 5
```

For notebooks, start Jupyter through the locked environment:

```bash
uv run --with jupyterlab jupyter lab
```

### Pull requests and CI

The workflow at `.github/workflows/ci.yml` runs Python tests and static checks,
builds platform wheels containing the Rust extension, and builds a source
distribution. Changes to Python dispatch code should be checked in both a
machine with the native extension available and a fallback environment where
`pyroads._native` cannot be imported.

Before opening a pull request, run:

```bash
uv sync --dev
uv run pytest
uv run ruff check src
uv run pyright src
cargo fmt --manifest-path rust/_native/Cargo.toml -- --check
cargo check --manifest-path rust/_native/Cargo.toml
uv build --out-dir /tmp/pyroads-dist
```

## Acknowledgements

The original `pyroads` package was created by Shaan Ciantar. Nicholas Archer
wrote most of the functionality in the former `merge-segments`,
`homogeneous-segmentation`, and `segmenter` packages.

Dagmawi Tadesse consolidated these packages into the `pyroads` repository,
preserving their public APIs for backward compatibility and delivering 
performance improvements through NumPy and Numba JIT compilation.

## Disclaimer

This software is made available as an open-source project by Main Roads
Western Australia.

The software is provided for general use and research purposes. Users are
responsible for validating the suitability of the software for their own
applications.

Main Roads Western Australia does not accept responsibility for decisions,
analyses, asset management outcomes, or other results produced using this
software.