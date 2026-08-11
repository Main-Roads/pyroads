from __future__ import annotations

import re

import pandas as pd

from ..pivot.cway_to_side import cway_to_side
from ..pivot.lane_to_row import lane_to_row as _lane_to_row
from .lane import get_lanes


def lane_to_side(value):
    values = pd.Series(value) if not isinstance(value, pd.Series) else value
    result = values.astype(str).str.extract(r"^([LR])", expand=False)
    return result if isinstance(value, pd.Series) else result.iloc[0]


def hsd_to_side(data, hsd):
    result = data.copy()
    result["side"] = result[hsd].map({"L": "L", "R": "R", "l": "L", "r": "R"})
    return result


def lane_to_row(
    data: pd.DataFrame,
    dirn: str,
    id_vars: list[str],
    start: str | None = None,
    end: str | None = None,
    start_true: str | None = None,
    end_true: str | None = None,
    prefixes: list[str] | str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Compatibility wrapper for the historical cross-sectional pivot API."""
    if prefixes is None:
        return _lane_to_row(
            data=data,
            id_vars=id_vars,
            lane_var="LANE_NO",
            side=dirn,
            start=start,
            end=end,
            start_true=start_true,
            end_true=end_true,
            prefixes=prefixes,
            **kwargs,
        )

    prefixes = [prefixes] if isinstance(prefixes, str) else prefixes
    measure_columns = [column for column in (start, end, start_true, end_true) if column]
    lane_numbers = sorted({
        match.group(1)
        for prefix in prefixes
        for column in data.columns
        if column.startswith(prefix)
        for match in [re.search(r"(\d+)$", column)]
        if match is not None
    }, key=int)
    rows = []
    for lane_number in lane_numbers:
        frame = data[id_vars + measure_columns].copy()
        frame["XSP"] = data[dirn].astype(str).to_numpy() + lane_number
        for prefix in prefixes:
            column = f"{prefix}{lane_number}"
            frame[prefix] = data[column].to_numpy() if column in data else float("nan")
        rows.append(frame)

    if not rows:
        return data.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)[id_vars + measure_columns + ["XSP", *prefixes]]


__all__ = ["cway_to_side", "hsd_to_side", "lane_to_side", "get_lanes", "lane_to_row"]
