"""Test verbose parameter functionality"""

import pandas as pd
import pyroads.merge.merge as merge
from io import StringIO
import sys


def test_verbose_false_suppresses_output():
    """Test that verbose=False suppresses diagnostic output"""

    segmentation = pd.DataFrame(
        columns=["road_no", "carriageway", "slk_from", "slk_to"],
        data=[
            ["H001", "L", 10, 50],
            ["H001", "L", 50, 100],
        ],
    )

    pavement_data = pd.DataFrame(
        columns=[
            "road_no",
            "carriageway",
            "slk_from",
            "slk_to",
            "pavement_width",
            "pavement_type",
        ],
        data=[
            ["H001", "L", 10, 20, 4.00, "tA"],
            ["H001", "L", 20, 40, 3.50, "tB"],
            ["H001", "L", 40, 80, 3.80, "tC"],
        ],
    )

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    # Run with verbose=False (default) - should show progress bar but not diagnostic messages
    result = merge.on_slk_intervals(
        target=segmentation,
        data=pavement_data,
        join_left=["road_no", "carriageway"],
        column_actions=[
            merge.Action("pavement_width", merge.Aggregation.LengthWeightedAverage()),
            merge.Action("pavement_type", merge.Aggregation.KeepLongest()),
        ],
        from_to=("slk_from", "slk_to"),
        legacy=False,
        verbose=False,
    )

    sys.stdout = old_stdout
    output = captured_output.getvalue()

    # Should not contain diagnostic messages about paths or fallbacks
    assert "[pyroads.merge] Using optimized path" not in output
    assert "[pyroads.merge] Falling back to categorical path" not in output

    # Result should still be valid
    assert len(result) == 2
    assert "pavement_width" in result.columns
    assert "pavement_type" in result.columns


def test_verbose_true_shows_output():
    """Test that verbose=True shows diagnostic output"""

    segmentation = pd.DataFrame(
        columns=["road_no", "carriageway", "slk_from", "slk_to"],
        data=[
            ["H001", "L", 10, 50],
            ["H001", "L", 50, 100],
        ],
    )

    pavement_data = pd.DataFrame(
        columns=[
            "road_no",
            "carriageway",
            "slk_from",
            "slk_to",
            "pavement_width",
            "pavement_type",
        ],
        data=[
            ["H001", "L", 10, 20, 4.00, "tA"],
            ["H001", "L", 20, 40, 3.50, "tB"],
            ["H001", "L", 40, 80, 3.80, "tC"],
        ],
    )

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    # Run with verbose=True - should show diagnostic messages
    result = merge.on_slk_intervals(
        target=segmentation,
        data=pavement_data,
        join_left=["road_no", "carriageway"],
        column_actions=[
            merge.Action("pavement_width", merge.Aggregation.LengthWeightedAverage()),
            merge.Action("pavement_type", merge.Aggregation.KeepLongest()),
        ],
        from_to=("slk_from", "slk_to"),
        legacy=False,
        verbose=True,
    )

    sys.stdout = old_stdout
    output = captured_output.getvalue()

    # Categorical actions use the optimized fallback path.
    assert "[pyroads.merge] Falling back to categorical path" in output
    assert "action(s)" in output
    assert "group(s)" in output

    # Result should still be valid
    assert len(result) == 2
    assert "pavement_width" in result.columns
    assert "pavement_type" in result.columns


def test_verbose_legacy_path():
    """Test that verbose parameter works with legacy path"""

    segmentation = pd.DataFrame(
        columns=["road_no", "carriageway", "slk_from", "slk_to"],
        data=[
            ["H001", "L", 10, 50],
        ],
    )

    pavement_data = pd.DataFrame(
        columns=["road_no", "carriageway", "slk_from", "slk_to", "pavement_width"],
        data=[
            ["H001", "L", 10, 20, 4.00],
        ],
    )

    # Legacy path should not crash with verbose parameter
    result = merge.on_slk_intervals(
        target=segmentation,
        data=pavement_data,
        join_left=["road_no", "carriageway"],
        column_actions=[
            merge.Action("pavement_width", merge.Aggregation.LengthWeightedAverage()),
        ],
        from_to=("slk_from", "slk_to"),
        legacy=True,
        verbose=False,
    )

    assert len(result) == 1
    assert "pavement_width" in result.columns
