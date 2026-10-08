"""Guard against the Polars merge path falling far behind the pandas path."""

import importlib.util
from pathlib import Path

import pytest

from pyroads import backend

pytest.importorskip("polars")

BENCHMARK = Path(__file__).resolve().parents[2] / "examples" / "merge" / "benchmark_backends.py"


@pytest.mark.slow
@pytest.mark.skipif(backend() != "rust", reason="timing guard targets the native backend")
def test_polars_merge_is_not_much_slower_than_pandas():
    spec = importlib.util.spec_from_file_location("benchmark_backends", BENCHMARK)
    assert spec is not None and spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    for label, pandas_time, polars_time in benchmark.compare(roads=300, repeats=5):
        # The fixed allowance absorbs scheduler noise on shared CI runners.
        assert polars_time < 1.5 * pandas_time + 0.05, (
            f"{label}: polars {polars_time:.3f}s vs pandas {pandas_time:.3f}s"
        )
