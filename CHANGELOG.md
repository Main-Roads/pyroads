# Changelog

## 0.5.0 - 2026-09-01

### Added

- Added Rayon-backed Rust kernels for the heaviest segmentation, reshaping, and
  interval merge operations.
- Added native extension packaging for GitHub source installs and platform
  wheels.

### Changed

- Rust is now used automatically when the native extension is available, with
  the existing Numba/Python implementations retained as a fallback.
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
- Added optional Polars and Dask merge backends.
- Added consolidated tests, examples, documentation images, and benchmarks.

### Changed

- Moved the former standalone APIs under the `pyroads` namespace while keeping
  compatibility options where practical.
- The optimized interval merge is now the default; `legacy=True` remains
  available for compatibility.
- Numba is a required dependency for the default implementation.
