"""Parity tests for the grouped merge shared by the pandas and Polars backends."""

import contextlib
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from pyroads.merge import Action, Aggregation, on_slk_intervals
from pyroads.merge import _grouped, _numba_merge

pl = pytest.importorskip("polars")

from pyroads.merge._polars_merge import on_slk_intervals_polars  # noqa: E402

FROM_TO = ("slk_from", "slk_to")
JOIN = ["road", "cwy"]

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    AGGREGATIONS = [
        Aggregation.LengthWeightedAverage(),
        Aggregation.Average(),
        Aggregation.First(),
        Aggregation.Sum(),
        Aggregation.Min(),
        Aggregation.Max(),
        Aggregation.IndexOfMin(),
        Aggregation.IndexOfMax(),
        Aggregation.SumProportionOfData(),
        Aggregation.SumProportionOfTarget(),
        Aggregation.KeepLongest(),
        Aggregation.KeepLongestSegment(),
        Aggregation.LengthWeightedPercentile(0.75),
    ]
ACTIONS = [
    Action("value", aggregation, f"value_{index}")
    for index, aggregation in enumerate(AGGREGATIONS)
] + [Action("surface", Aggregation.KeepLongest(), "surface_kl")]


def make_frames(roads=6, targets_per_road=40, data_per_road=150, seed=7):
    """Shuffled multi-key frames with ties, duplicates, unmatched and missing keys."""
    rng = np.random.default_rng(seed)
    target_rows, data_rows = [], []
    for road in range(roads):
        for cwy in ("L", "R"):
            for position in range(targets_per_road):
                target_rows.append((f"R{road}", cwy, position * 10.0, position * 10.0 + 10.0))
            starts = rng.integers(0, targets_per_road * 10 - 5, data_per_road).astype(float)
            ends = starts + rng.integers(1, 25, data_per_road)
            for start, end in zip(starts, ends):
                data_rows.append(
                    (f"R{road}", cwy, start, end, float(rng.integers(0, 5)), rng.choice(["AC", "SS", None]))
                )
            # Duplicate interval and equal-overlap tie with different categories.
            data_rows.append((f"R{road}", cwy, 0.0, 5.0, 1.0, "AC"))
            data_rows.append((f"R{road}", cwy, 5.0, 10.0, 2.0, "SS"))
            data_rows.append((f"R{road}", cwy, 0.0, 5.0, 3.0, "PM"))
    target_rows += [("TARGET_ONLY", "L", 0.0, 10.0), (None, "L", 0.0, 10.0)]
    data_rows += [("DATA_ONLY", "L", 0.0, 10.0, 1.0, "AC"), (None, "L", 0.0, 10.0, 9.0, "PM")]
    target = pd.DataFrame(target_rows, columns=[*JOIN, *FROM_TO])
    data = pd.DataFrame(data_rows, columns=[*JOIN, *FROM_TO, "value", "surface"])
    return (
        target.sample(frac=1, random_state=1).reset_index(drop=True),
        data.sample(frac=1, random_state=2).reset_index(drop=True),
    )


def to_polars(frame):
    return pl.DataFrame(
        {
            column: [None if pd.isna(value) else value for value in frame[column].tolist()]
            for column in frame.columns
        }
    )


def assert_frames_match(pandas_result, polars_result, actions=ACTIONS):
    for action in actions:
        expected = pandas_result[action.rename].to_numpy()
        actual = polars_result[action.rename].to_numpy()
        if action.rename == "surface_kl":
            assert list(actual) == [None if pd.isna(value) else value for value in expected]
        else:
            np.testing.assert_array_equal(actual, expected.astype(float), err_msg=action.rename)


def without_grouped_kernel():
    native = _numba_merge._rust_native
    names = [name for name in dir(native) if not name.startswith("_") and name != "merge_groups"]
    return types.SimpleNamespace(**{name: getattr(native, name) for name in names})


def test_pandas_and_polars_match():
    target, data = make_frames()
    pandas_result = on_slk_intervals(target, data, JOIN, ACTIONS, FROM_TO)
    polars_result = on_slk_intervals(to_polars(target), to_polars(data), JOIN, ACTIONS, FROM_TO)
    assert_frames_match(pandas_result, polars_result)


def test_unmatched_and_missing_keys_are_empty():
    target, data = make_frames()
    for result in (
        on_slk_intervals(target, data, JOIN, ACTIONS, FROM_TO),
        on_slk_intervals(to_polars(target), to_polars(data), JOIN, ACTIONS, FROM_TO).to_pandas(),
    ):
        unmatched = result["road"].isna() | (result["road"] == "TARGET_ONLY")
        assert unmatched.sum() == 2
        assert result.loc[unmatched, "value_0"].isna().all()
        assert result.loc[unmatched, "surface_kl"].isna().all()
        assert result.loc[~unmatched, "value_0"].notna().any()


def test_keep_longest_tie_goes_to_first_seen():
    target = pd.DataFrame({"road": ["A"], "slk_from": [0.0], "slk_to": [10.0]})
    data = pd.DataFrame(
        {
            "road": ["A", "A", "A"],
            "slk_from": [0.0, 5.0, 0.0],
            "slk_to": [5.0, 10.0, 5.0],
            "surface": ["SS", "AC", "SS"],
        }
    )
    data = data.iloc[[1, 0, 2]].reset_index(drop=True)
    actions = [Action("surface", Aggregation.KeepLongest(), "surface_kl")]
    pandas_result = on_slk_intervals(target, data.iloc[:2], ["road"], actions, FROM_TO)
    polars_result = on_slk_intervals_polars(
        to_polars(target), to_polars(data.iloc[:2]), ["road"], actions, FROM_TO
    )
    assert pandas_result["surface_kl"].tolist() == polars_result["surface_kl"].to_list()
    # Totals win over order once the overlap is no longer tied.
    assert on_slk_intervals(target, data, ["road"], actions, FROM_TO)["surface_kl"].tolist() == ["SS"]


@pytest.mark.parametrize("native", ["stale", "missing"])
def test_fallback_paths_match(monkeypatch, native):
    target, data = make_frames(roads=3)
    reference = on_slk_intervals(target, data, JOIN, ACTIONS, FROM_TO)
    replacement = without_grouped_kernel() if native == "stale" else None
    monkeypatch.setattr(_numba_merge, "_rust_native", replacement)
    with pytest.warns(RuntimeWarning) if native == "missing" else contextlib.nullcontext():
        monkeypatch.setattr("pyroads._backend._fallback_announced", False)
        pandas_result = on_slk_intervals(target, data, JOIN, ACTIONS, FROM_TO)
    polars_result = on_slk_intervals(to_polars(target), to_polars(data), JOIN, ACTIONS, FROM_TO)
    assert_frames_match(pandas_result, polars_result)
    if native == "stale":
        assert_frames_match(reference, polars_result)


def test_thread_count_does_not_change_results():
    target, data = make_frames(roads=8, targets_per_road=600, data_per_road=900)
    target_pl, data_pl = to_polars(target), to_polars(data)
    results = [
        on_slk_intervals_polars(target_pl, data_pl, JOIN, ACTIONS, FROM_TO, n_jobs=n_jobs)
        for n_jobs in (None, 1, 3)
    ]
    for result in results[1:]:
        assert result.equals(results[0])


def test_invalid_thread_count_raises():
    target, data = make_frames(roads=1)
    with pytest.raises(ValueError):
        on_slk_intervals_polars(to_polars(target), to_polars(data), JOIN, ACTIONS, FROM_TO, n_jobs=0)


def test_pandas_group_ids_mark_missing_keys():
    target = pd.DataFrame({"road": ["A", None, "B"], "cwy": ["L", "L", None]})
    data = pd.DataFrame({"road": ["B", "A"], "cwy": ["L", "L"]})
    target_ids, data_ids = _grouped.pandas_group_ids(target, data, ["road", "cwy"])
    assert target_ids.tolist() == [0, -1, -1]
    assert data_ids.tolist() == [1, 0]
