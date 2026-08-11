import pandas as pd
import pytest

from pyroads.merge import merge
from pyroads.merge.exceptions import InvalidAggregationError


def _string_value_frames():
    target = pd.DataFrame(
        [["R1", 0, 10]],
        columns=["road", "slk_from", "slk_to"],
    )
    data = pd.DataFrame(
        [["R1", 0, 10, "x"], ["R1", 0, 10, "y"]],
        columns=["road", "slk_from", "slk_to", "label"],
    )
    return target, data


@pytest.mark.parametrize(
    ("aggregation_factory", "aggregation_name"),
    [
        (merge.Aggregation.Sum, "Sum"),
        (merge.Aggregation.Average, "Average"),
    ],
)
@pytest.mark.parametrize(
    "runner_name",
    ["on_slk_intervals", "on_slk_intervals_optimized", "on_slk_intervals_fallback"],
)
def test_non_numeric_aggregations_are_rejected(
    aggregation_factory, aggregation_name, runner_name
):
    target, data = _string_value_frames()
    runner = getattr(merge, runner_name)

    with pytest.raises(
        InvalidAggregationError,
        match=(
            f"Aggregation '{aggregation_name}' requires numeric data in column 'label'"
        ),
    ):
        runner(
            target=target,
            data=data,
            join_left=["road"],
            column_actions=[
                merge.Action(
                    "label",
                    aggregation=aggregation_factory(),
                    rename="label_result",
                )
            ],
            from_to=("slk_from", "slk_to"),
        )


@pytest.mark.skipif(
    not merge.is_numba_available(),
    reason="Numba not installed",
)
def test_non_numeric_aggregations_are_rejected_numba():
    target, data = _string_value_frames()

    with pytest.raises(
        InvalidAggregationError,
        match="Aggregation 'Sum' requires numeric data in column 'label'",
    ):
        merge.on_slk_intervals_numba(
            target=target,
            data=data,
            join_left=["road"],
            column_actions=[
                merge.Action(
                    "label",
                    aggregation=merge.Aggregation.Sum(),
                    rename="label_result",
                )
            ],
            from_to=("slk_from", "slk_to"),
        )


def test_on_slk_intervals_auto_uses_numba_when_available(monkeypatch):
    sentinel = pd.DataFrame([[1.0]], columns=["value_sum"])

    def fake_numba(**kwargs):
        return sentinel

    def fail_optimized(**kwargs):
        raise AssertionError(
            "optimized path should not be used when numba is available"
        )

    monkeypatch.setattr(merge, "is_numba_available", lambda: True)
    monkeypatch.setattr(merge, "on_slk_intervals_numba", fake_numba)
    monkeypatch.setattr(merge, "on_slk_intervals_optimized", fail_optimized)

    result = merge.on_slk_intervals_auto(
        target=pd.DataFrame(),
        data=pd.DataFrame(),
        join_left=[],
        column_actions=[],
        from_to=("slk_from", "slk_to"),
        prefer_optimized=True,
    )

    assert result is sentinel


def test_on_slk_intervals_legacy_false_propagates_numba_errors(monkeypatch):
    def fail_numba(**kwargs):
        raise RuntimeError("boom")

    def fail_optimized(**kwargs):
        raise AssertionError("optimized fallback should not swallow numba errors")

    monkeypatch.setattr(merge, "is_numba_available", lambda: True)
    monkeypatch.setattr(merge, "on_slk_intervals_numba", fail_numba)
    monkeypatch.setattr(merge, "on_slk_intervals_optimized", fail_optimized)

    with pytest.raises(RuntimeError, match="boom"):
        merge.on_slk_intervals(
            target=pd.DataFrame(),
            data=pd.DataFrame(),
            join_left=[],
            column_actions=[],
            from_to=("slk_from", "slk_to"),
            legacy=False,
        )
