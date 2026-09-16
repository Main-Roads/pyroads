import pytest

from pyroads.mappings.surface import surf_type
from pyroads.reshape import make_segments
from pyroads.segmenter import split_rows_by_category_to_max_segment_length

pl = pytest.importorskip("polars")


def test_segmenter_accepts_and_returns_polars():
    data = pl.DataFrame(
        {
            "road": ["R1"],
            "slk_from": [0.0],
            "slk_to": [10.0],
            "true_from": [0.0],
            "true_to": [10.0],
            "value": [7],
        }
    )

    result = split_rows_by_category_to_max_segment_length(
        data,
        ("slk_from", "slk_to"),
        ("true_from", "true_to"),
        ["road"],
        5.0,
    )

    assert isinstance(result, pl.DataFrame)
    assert result.height == 2


def test_reshape_accepts_and_returns_polars():
    data = pl.DataFrame(
        {"road": ["R1"], "START_SLK": [12.2], "END_SLK": [12.4]}
    )

    result = make_segments(data, start="START_SLK", end="END_SLK", max_segment=100)

    assert isinstance(result, pl.DataFrame)
    assert result.height == 2


def test_mapping_accepts_polars():
    data = pl.DataFrame({"surface_type": [1, 4]})

    result = surf_type(data, source="surface_type", to="short")

    assert isinstance(result, pl.Series)
    assert result.to_list() == ["DGA", "Concrete"]