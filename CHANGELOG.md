# Changelog

## 0.8.0 - 2026-10-08

### Performance

- Interval merges now process every join group in one native call that
  releases the GIL and runs groups in parallel. On a 4,000-group benchmark
  (200k target rows, 800k data rows) Polars inputs went from 16.7s to 0.12s
  and pandas inputs from 2.9s to 0.12s; with a categorical `KeepLongest`
  action, from 22.7s to 0.17s (Polars) and 9.0s to 0.15s (pandas).
- Polars string columns are factorized natively for `KeepLongest`.
- Native kernels run serially on small inputs instead of waking the Rayon
  thread pool for every call.

### Changed

- Polars merges no longer match rows whose join key is missing, consistent
  with the pandas backend.
- `on_slk_intervals_polars(n_jobs=...)` now sizes the native thread pool; the
  Python thread pool was removed. Non-positive values raise `ValueError`.
- The Numba fallback notice is a one-time `RuntimeWarning` instead of a
  message printed on every call.

### Added

- `examples/merge/benchmark_backends.py` and a test that fails when Polars
  merges are more than 1.5 times slower than pandas.
- CI runs the Polars tests, Rust unit tests, and `cargo fmt --check`.

## 0.7.0 - 2026-09-29

### Performance

- Fused numeric interval overlap and aggregation in Rust.
- Reused categorical factorization across merge groups.
- Removed an extra native round trip from cross-section processing.
- Fused homogeneous bisection and stretch row expansion in Rust.

## 0.6.0 - 2026-09-16

### Changed

- Added native Rust routing for numeric and categorical interval merge actions.
- Added Polars support across the public dataframe APIs.
- Removed the Dask backend.

## 0.5.0 - 2026-09-01

### Added

- Added Rayon-backed Rust kernels for the heaviest segmentation, reshaping, and
  interval merge operations.
- Added native extension packaging for GitHub source installs and platform
  wheels.
- Added automated GitHub Release wheels for Windows x86-64 and Linux x86-64.

### Changed

- Rust is now used automatically when the native extension is available, with
  the existing Numba/Python implementations retained as a fallback.
- Added `pyroads.backend()` to report the active Rust or Python backend.
- Added developer documentation for building, testing, linting, and releasing
  the project.

## 0.4.0 - 2026-08-11

### Added

- Consolidated four Main Roads WA Python packages into `pyroads`:
  - `pyroads` road-data utilities;
  - `segmenter`;
  - `homogeneous-segmentation`;
  - `merge-segments`.
- Added public modules for segmentation, homogeneous segmentation, interval
  merging, mappings, reshaping, and road-network data retrieval.
- Added Numba-backed interval and segmentation kernels, with optimized merge
  and cross-section paths.
- Added an optional Polars merge backend.
- Added consolidated tests, examples, documentation images, and benchmarks.

### Changed

- Moved the former standalone APIs under the `pyroads` namespace while keeping
  compatibility options where practical.
- The optimized interval merge is now the default; `legacy=True` remains
  available for compatibility.
- Numba is a required dependency for the default implementation.
