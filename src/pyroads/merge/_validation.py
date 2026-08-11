"""Validation utilities shared by optimized merge helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

if TYPE_CHECKING:
    from .merge import Action

from .exceptions import (
    DuplicateLabelError,
    InvalidAggregationError,
    InvalidDataFrameError,
    InvalidJoinConfigurationError,
    OutputCollisionError,
    ZeroLengthSegmentError,
)


_NUMERIC_ONLY_AGGREGATIONS = frozenset(
    {
        "Average",
        "LengthWeightedAverage",
        "LengthWeightedPercentile",
        "SumProportionOfData",
        "SumProportionOfTarget",
        "Sum",
        "IndexOfMax",
        "IndexOfMin",
        "Min",
        "Max",
    }
)


def ensure_dataframe(name: str, frame: object) -> pd.DataFrame:
    """Return *frame* if it is a DataFrame, otherwise raise.

    Args:
            name: Human-friendly name used in error messages.
            frame: Object to validate.

    Raises:
            InvalidDataFrameError: If *frame* is not a :class:`pandas.DataFrame`.
    """

    if not isinstance(frame, pd.DataFrame):
        raise InvalidDataFrameError(
            f"`{name}` parameter must be a pandas dataframe, received {type(frame)}"
        )
    return frame


def ensure_simple_index(name: str, frame: pd.DataFrame) -> None:
    """Validate that *frame* does not use MultiIndex structures or duplicates."""

    if isinstance(frame.index, pd.MultiIndex):
        raise InvalidDataFrameError(
            f"the `{name}` dataframe uses a `pandas.MultiIndex` which is not supported. "
            "Please call `.reset_index()`."
        )
    if isinstance(frame.columns, pd.MultiIndex):
        raise InvalidDataFrameError(
            f"the `{name}` dataframe uses a `pandas.MultiIndex` for columns which is "
            "not supported."
        )
    if frame.index.has_duplicates:
        raise DuplicateLabelError(
            f"`{name}` dataframe has duplicated index values. Please call `reset_index()`."
        )
    if frame.columns.has_duplicates:
        raise DuplicateLabelError(f"`{name}` dataframe has duplicated column names.")


def ensure_required_columns(
    target: pd.DataFrame,
    data: pd.DataFrame,
    join_left: Sequence[str],
    interval: Tuple[str, str],
) -> None:
    """Ensure both frames contain required join/interval columns."""

    required: List[str] = [*join_left, *interval]
    missing_messages: List[str] = []
    for column_name in required:
        in_target = column_name in target.columns
        in_data = column_name in data.columns
        if not in_target and not in_data:
            missing_messages.append(
                f"Column '{column_name}' is missing from both `target` and `data`."
            )
        elif not in_target:
            missing_messages.append(f"Column '{column_name}' is missing from `target`.")
        elif not in_data:
            missing_messages.append(f"Column '{column_name}' is missing from `data`.")
    if missing_messages:
        raise InvalidJoinConfigurationError(
            "Please check the `join_left` and `from_to` parameters."
            "Specified columns must be present and have matching names in both "
            "`target` and `data`:\n" + "\n".join(missing_messages)
        )


def ensure_nonzero_lengths(
    name: str,
    frame: pd.DataFrame,
    from_col: str,
    to_col: str,
) -> None:
    """Ensure that *frame* contains no zero-length intervals."""

    if (frame[from_col] == frame[to_col]).any():
        raise ZeroLengthSegmentError(
            f"`{name}` dataframe has rows where {from_col} == {to_col}. "
            "The merge tool does not work with zero length segments."
        )


def ensure_output_columns_available(
    target_columns: Iterable[str],
    actions: Sequence["Action"],
) -> None:
    """Ensure aggregated column names will not collide with target columns."""

    target_set = set(target_columns)
    for action in actions:
        rename = action.rename
        if rename in target_set:
            if rename == action.column_name:
                raise OutputCollisionError(
                    "Cannot merge column "
                    f"'{action.column_name}' into target because the target already "
                    "contains a column of that name. Please consider using "
                    "the rename parameter; `Action(..., rename='xyz')`."
                )
            else:
                raise OutputCollisionError(
                    "Cannot merge column "
                    f"'{action.column_name}' as '{rename}' into target because the "
                    f"target already contains a column named '{rename}'."
                )


def ensure_aggregation_column_types(
    data: pd.DataFrame,
    actions: Sequence["Action"],
) -> None:
    """Ensure aggregation strategies are compatible with source dtypes."""

    invalid_messages: List[str] = []
    for action in actions:
        column_name = action.column_name
        if column_name not in data.columns:
            continue

        aggregation_name = action.aggregation.type.name
        if aggregation_name not in _NUMERIC_ONLY_AGGREGATIONS:
            continue

        series = data[column_name]
        if not is_numeric_dtype(series):
            invalid_messages.append(
                "Aggregation "
                f"'{aggregation_name}' requires numeric data in column '{column_name}' "
                f"(dtype: {series.dtype})."
            )

    if invalid_messages:
        raise InvalidAggregationError("\n".join(invalid_messages))
