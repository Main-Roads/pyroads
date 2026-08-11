import os
import time
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, cast

from deprecated import deprecated

import numpy as np
import pandas
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import _validation as validation
from .exceptions import PercentileConfigurationError

# Try to import Numba-accelerated implementation
try:
    from ._numba_merge import on_slk_intervals_numba, is_numba_available

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def is_numba_available() -> bool:
        """Check if Numba is available for acceleration."""
        return False

    def on_slk_intervals_numba(
        target: Any,
        data: Any,
        join_left: List[str],
        column_actions: List[Any],
        from_to: Tuple[str, str],
        verbose: bool = False,
    ) -> Any:
        """Fallback stub when Numba is not installed."""
        raise ImportError(
            "Numba is required for on_slk_intervals_numba. "
            "Install with: pip install pyroads"
        )


_T = TypeVar("_T")

PerformanceMetrics = Dict[str, float]
PerformanceCallback = Callable[[str, PerformanceMetrics], None]

_performance_logger: Optional[PerformanceCallback] = None


def configure_performance_logger(callback: Optional[PerformanceCallback]) -> None:
    """Register a callback to receive merge performance metrics.

    Pass ``None`` to disable logging. The callback receives an event label and a
    dictionary containing numeric metrics captured for the completed merge.
    """

    global _performance_logger
    _performance_logger = callback


def _emit_performance_event(event: str, **metrics: float) -> None:
    if _performance_logger is not None:
        _performance_logger(
            event, {key: float(value) for key, value in metrics.items()}
        )


def _configure_default_performance_logger_from_env() -> None:
    mode = os.getenv("MERGE_SEGMENTS_PERF_LOG")
    if mode and mode.lower() == "stdout":

        def _stdout_logger(event: str, log_metrics: PerformanceMetrics) -> None:
            ordered = ", ".join(
                f"{key}={value:.6f}" for key, value in sorted(log_metrics.items())
            )
            print(f"[pyroads.merge][perf] {event}: {ordered}")

        configure_performance_logger(_stdout_logger)


try:
    from tqdm import tqdm as _real_tqdm  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover - optional dependency
    _tqdm_callable: Optional[Callable[..., Iterable[Any]]] = None
else:
    _tqdm_callable = _real_tqdm


def tqdm(iterable: Iterable[_T], **kwargs: Any) -> Iterable[_T]:
    """Return the tqdm iterator when available, otherwise passthrough."""
    if _tqdm_callable is None:
        return iterable
    return cast(Iterable[_T], _tqdm_callable(iterable, **kwargs))


class AggregationType(Enum):
    KeepLongestSegment = 1  # Deprecated
    KeepLongest = 2
    Average = 3
    LengthWeightedAverage = 4
    LengthWeightedPercentile = 5
    First = 6
    SumProportionOfData = 7
    SumProportionOfTarget = 8
    Sum = 9
    IndexOfMax = 10
    IndexOfMin = 11
    Min = 12
    Max = 13


class Aggregation:
    def __init__(
        self, aggregation_type: AggregationType, percentile: Optional[float] = None
    ):
        """Don't initialise this class directly, please use one of the static factory functions"""
        self.type: AggregationType = aggregation_type
        self.percentile: Optional[float] = percentile
        pass

    @staticmethod
    def First():
        return Aggregation(AggregationType.First)

    @staticmethod
    @deprecated(
        version="0.1.0",
        reason="`merge.Aggregation.KeepLongestSegment()` is an old, incorrect implementation. Please use `merge.Aggregation.KeepLongest()`",
    )
    def KeepLongestSegment():
        """
        DEPRECATED: Please use `KeepLongest()` instead.
        """
        # print("WARNING `KeepLongestSegment` is deprecated, please use `KeepLongest` instead.\n`KeepLongestSegment` kept here temporarily for testing purposes but is will be removed in future versions.")
        return Aggregation(AggregationType.KeepLongestSegment)

    @staticmethod
    def KeepLongest():
        return Aggregation(AggregationType.KeepLongest)

    @staticmethod
    def LengthWeightedAverage():
        return Aggregation(AggregationType.LengthWeightedAverage)

    @staticmethod
    def Average():
        """
        The average non-blank overlapping value. If all overlapping values are blank, keep blank.
        """
        return Aggregation(AggregationType.Average)

    @staticmethod
    def LengthWeightedPercentile(percentile: float):
        """
        The length weighted percentile of overlapping values.
        This is similar to a normal percentile calculation, but the length of the overlapping segment is taken into account.

        There is a complicated sort-by-Value, followed by an interpolation step.
        In the following diagram the `▴` shows the value of the 75th percentile.
        The `▴` is at 75% of the total length (chainage/SLK length) of all overlapping values
        (not including the length of half the first segment and half the last segment)
        and is interpolated between center-point values (`○`) of the last two segments:

        ```text
              |                          _○_
              |                         |   |
              |                      ▴  |   |   <---- 75th percentile value
        Value |              _____○_____|   |
              |        __○__|           |   |
              |       |     |           |   |
              |  __○__|     |           |   |
              | |     |     |           |   |
                   |<-----SLK Length----->|
                   0%                ↑   100%
                                     │
              75th percentile ───────┘
        ```
        """
        if percentile > 1.0 or percentile < 0.0:
            raise ValueError(
                f"Percentile out of range. Must be greater than 0.0 and less than 1.0. Got {percentile}."
                + (" Do you need to divide by 100?" if percentile > 1.0 else "")
            )
        return Aggregation(
            AggregationType.LengthWeightedPercentile, percentile=percentile
        )

    @staticmethod
    @deprecated(
        version="0.4.3",
        reason="Aggregation type is renamed; Please use `Aggregation.SumProportionOfData()` for equivalent behaviour.",
    )
    def ProportionalSum():
        """
        DEPRECATED: Please use `merge.Aggregation.SumProportionOfData()` for equivalent behaviour. `ProportionalSum()` will be removed in the future.

        The sum of all overlapping `data` segments,
        where the value of each overlapping segment is multiplied by
        the length of the overlap divided by the length of the `data` segment.
        This is the same behaviour as the old VBA macro.
        See also `SumProportionOfTarget()`
        """
        return Aggregation(AggregationType.SumProportionOfData)

    @staticmethod
    def SumProportionOfTarget():
        """
        The sum of all overlapping `data` segments,
        where the value of each overlapping segment is multiplied by
        the length of the overlap divided by the length of the `target` segment.

        This aggregation method is suitable when aggregating columns measured in
        `Units per Kilometre` or `% of length`. The aggregated value will have the same unit.
        The assumption is that the % of length is spread evenly across the whole data segment.
        (This aggregation was created to deal with the cracking dataset which is given in 10 metre segments with a % cracked.
        Note that a better aggregation should be used if there is concern regarding the relative width between the `data` and `target` segments)

        The result below is calculated as `result = (20%*10 + 40%*5)/40 = 10%`

        ```text
        data   :   |-- len=20---value=20% --|                 |----------------------- len=50 --- value=40% ----------------|
        target :              |------------- len=40 ------------------|
        overlap:              |--- len=10 --|                 |-len=5-|
        result :              |------------ value=10% ----------------|
        ```

        See also `SumProportionOfData()`
        """
        return Aggregation(AggregationType.SumProportionOfTarget)

    @staticmethod
    def SumProportionOfData():
        """
        The sum of all overlapping `data` segments,
        where the value of each overlapping segment is multiplied by
        the length of the overlap divided by the length of the `data` segment.
        This is the same behaviour as the old VBA macro.
        See also `SumProportionOfTarget()`
        """
        return Aggregation(AggregationType.SumProportionOfData)

    @staticmethod
    def Sum():
        """This is the sum of values touching the target. Even if only part of the value is overlapping the target segment, the entire data value will be added to the sum"""
        return Aggregation(AggregationType.Sum)

    @staticmethod
    def IndexOfMax():
        """The index (or row label) of the `data` DataFrame, of the maximum overlapping segment"""
        return Aggregation(AggregationType.IndexOfMax)

    @staticmethod
    def IndexOfMin():
        """The index (or row label) of the `data` DataFrame, of the minimum overlapping segment"""
        return Aggregation(AggregationType.IndexOfMin)

    @staticmethod
    def Max():
        """Value of the maximum overlapping segment"""
        return Aggregation(AggregationType.Max)

    @staticmethod
    def Min():
        """Value of the minimum overlapping segment"""
        return Aggregation(AggregationType.Min)


class Action:
    def __init__(
        self, column_name: str, aggregation: Aggregation, rename: Optional[str] = None
    ):
        self.column_name: str = column_name
        self.rename = rename if rename is not None else self.column_name
        self.aggregation: Aggregation = aggregation


def _normalize_group_key(values) -> tuple:
    """Return deterministic tuple keys for groupby outputs."""
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(values)
    return (values,)


def _keep_longest_tie_first(values: Any, overlaps: Any) -> Any:
    """Return the value with the largest total overlap, preferring first-seen ties."""

    codes, uniques = pd.factorize(values, sort=False)
    if len(uniques) == 0:
        return np.nan
    totals = np.bincount(
        codes,
        weights=np.asarray(overlaps, dtype=float),
        minlength=len(uniques),
    )
    return uniques[int(totals.argmax())]


def _build_data_groups(
    data_subset: pd.DataFrame, join_left: List[str]
) -> Dict[tuple, pd.DataFrame]:
    groups: Dict[tuple, pd.DataFrame] = {}
    for key, group in data_subset.groupby(join_left, sort=False):
        groups[_normalize_group_key(key)] = group.reset_index(drop=True)
    return groups


def _should_use_categorical_fallback(
    data_subset: pd.DataFrame, column_actions: List[Action]
) -> List[str]:
    """Return column names that require categorical fallback."""
    needs_fallback: List[str] = []
    for action in column_actions:
        if action.aggregation.type == AggregationType.KeepLongest:
            series = data_subset[action.column_name]
            if not is_numeric_dtype(series):
                needs_fallback.append(action.column_name)
    return needs_fallback


def _validate_inputs(
    target: object,
    data: object,
    join_left: List[str],
    column_actions: List[Action],
    from_to: Tuple[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate common merge inputs and return DataFrame views."""
    if not isinstance(join_left, list):
        raise TypeError("`join_left` must be a list of column names.")

    target_df = validation.ensure_dataframe("target", target)
    data_df = validation.ensure_dataframe("data", data)

    validation.ensure_simple_index("target", target_df)
    validation.ensure_simple_index("data", data_df)
    validation.ensure_required_columns(target_df, data_df, join_left, from_to)

    slk_from, slk_to = from_to
    validation.ensure_nonzero_lengths("target", target_df, slk_from, slk_to)
    validation.ensure_nonzero_lengths("data", data_df, slk_from, slk_to)
    validation.ensure_output_columns_available(target_df.columns, column_actions)
    validation.ensure_aggregation_column_types(data_df, column_actions)

    return target_df, data_df


def on_slk_intervals_legacy(
    target: pd.DataFrame,
    data: pd.DataFrame,
    join_left: List[str],
    column_actions: List[Action],
    from_to: Tuple[str, str],
    verbose: bool = False,
):
    start_time = time.perf_counter()
    slk_from, slk_to = from_to

    result_index: list[Any] = []
    result_rows: list[list[Any]] = []

    target, data = _validate_inputs(target, data, join_left, column_actions, from_to)

    # ReIndex data for faster O(N) lookup
    data = data.assign(data_id=data.index)
    data = data.set_index([*join_left, "data_id"])
    data = data.sort_index()

    # Group target data by Road Number and Carriageway
    try:
        target_groups = target.groupby(join_left)
    except KeyError:
        matching_columns = [col for col in join_left if col in target.columns]
        raise Exception(
            f"Parameter join_left={join_left} did not match"
            + (
                " any columns in the target DataFrame"
                if len(matching_columns) == 0
                else f" all columns in target DataFrame. Only matched columns {matching_columns}"
            )
        )

    # Main Loop
    # TODO: address pandas warning regarding groupers with length of 1; will not return a tuple. Annoying, why?
    for target_group_index, target_group in target_groups:
        try:
            data_matching_target_group = data.loc[target_group_index]
        except KeyError:
            # There was no data matching the target group. Skip adding output. output to these rows will be NaN for all columns.
            continue
        except TypeError as e:
            # The datatype of group_index is picky... sometimes it wants a tuple, sometimes it will accept a list
            # this appears to be a bug or inconsistency with pandas when using multi-index dataframes.
            print(f"Error: Could not group the following data by {target_group_index}:")
            print(f"type(group_index)  {type(target_group_index)}:")
            print("the data:")
            print(data)
            raise e

        # Iterate row by row through the target group
        for target_index, target_row in target_group.iterrows():
            # Select data with overlapping slk interval
            data_to_aggregate_for_target_group = data_matching_target_group[
                (data_matching_target_group[slk_from] < target_row[slk_to])
                & (data_matching_target_group[slk_to] > target_row[slk_from])
            ]  # .copy()
            # TODO: the copy function on the line above has a lot to do with the slowness of this algorithm
            #       because all columns are copied, not just the ones we are aggregating, for wide dataframes
            #       there is potentially a huge amount of memory allocated and deallocated that doesnt need to be.
            #       only needs to be copied so that the "overlap_len" column can be added. If we can avoid adding
            #       this column we might do a lot better.

            # if no data matches the target group then skip
            if data_to_aggregate_for_target_group.empty:
                continue

            # compute overlaps for each row of data
            overlap_min = np.maximum(
                data_to_aggregate_for_target_group[slk_from], target_row[slk_from]
            )
            overlap_max = np.minimum(
                data_to_aggregate_for_target_group[slk_to], target_row[slk_to]
            )

            # overlap_len = np.maximum(overlap_max - overlap_min, 0)  # np.maximum() is not needed due to filters above
            overlap_len = overlap_max - overlap_min

            # expect this to trigger warning about setting value on view?
            # does not seem to though
            # data_to_aggregate_for_target_group["overlap_len"] = overlap_len  # Remove this... there is no reason to attached overlap_len to the original dataframe

            # for each column of data that we keep, we must aggregate each field down to a single value
            # create a blank row to store the result of each column
            aggregated_result_row: list[Any] = []
            for column_action_index, column_action in enumerate(column_actions):
                column_len_to_aggregate: Any = data_to_aggregate_for_target_group.loc[
                    :, [column_action.column_name]
                ].assign(
                    overlap_len=overlap_len
                )  # assign is done here so that NaN data can be dropped at the same time as the overlap lengths. Later we also benefit from the combination by being able to concurrently sort both columns.
                column_len_to_aggregate = column_len_to_aggregate[
                    ~column_len_to_aggregate.iloc[:, 0].isna()
                    & (column_len_to_aggregate["overlap_len"] > 0)
                ]

                if column_len_to_aggregate.empty:
                    # Infill with np.nan or we will lose our column position.
                    aggregated_result_row.append(np.nan)
                    continue

                column_to_aggregate: pandas.Series = column_len_to_aggregate.iloc[:, 0]
                column_to_aggregate_overlap_len: pandas.Series = (
                    column_len_to_aggregate.iloc[:, 1]
                )

                if column_action.aggregation.type == AggregationType.Average:
                    aggregated_result_row.append(column_to_aggregate.mean())

                elif column_action.aggregation.type == AggregationType.First:
                    aggregated_result_row.append(column_to_aggregate.iloc[0])

                elif (
                    column_action.aggregation.type
                    == AggregationType.LengthWeightedAverage
                ):
                    total_overlap_length = column_to_aggregate_overlap_len.sum()
                    aggregated_result_row.append(
                        (column_to_aggregate * column_to_aggregate_overlap_len).sum()
                        / total_overlap_length
                    )

                elif (
                    column_action.aggregation.type == AggregationType.KeepLongestSegment
                ):
                    aggregated_result_row.append(
                        column_to_aggregate.loc[
                            column_to_aggregate_overlap_len.idxmax()
                        ]
                    )

                elif column_action.aggregation.type == AggregationType.KeepLongest:
                    aggregated_result_row.append(
                        _keep_longest_tie_first(
                            column_to_aggregate,
                            column_to_aggregate_overlap_len,
                        )
                    )

                elif (
                    column_action.aggregation.type
                    == AggregationType.LengthWeightedPercentile
                ):
                    column_len_to_aggregate = column_len_to_aggregate.sort_values(
                        by=column_action.column_name, ascending=True
                    )

                    column_to_aggregate = column_len_to_aggregate.iloc[
                        :, 0
                    ]  # TODO: Why is this repeated?
                    column_to_aggregate_overlap_len = column_len_to_aggregate.iloc[
                        :, 1
                    ]  # TODO: Why is this repeated?

                    rolling_mean: Any = column_to_aggregate_overlap_len.rolling(2).mean()
                    x_coords: Any = rolling_mean.fillna(0).cumsum()
                    total = float(x_coords.iloc[-1])
                    if total <= 0:
                        aggregated_result_row.append(column_to_aggregate.iloc[-1])
                        continue
                    x_coords /= total
                    percentile = column_action.aggregation.percentile
                    if percentile is None:
                        raise ValueError(
                            "LengthWeightedPercentile requires a numeric percentile."
                        )
                    result = np.interp(
                        float(percentile), x_coords.to_numpy(), column_to_aggregate
                    )
                    aggregated_result_row.append(result)

                elif (
                    column_action.aggregation.type
                    == AggregationType.SumProportionOfData
                ):
                    data_to_aggregate_for_target_group_slk_length = (
                        data_to_aggregate_for_target_group[slk_to]
                        - data_to_aggregate_for_target_group[slk_from]
                    )
                    aggregated_result_row.append(
                        (
                            column_to_aggregate
                            * column_to_aggregate_overlap_len
                            / data_to_aggregate_for_target_group_slk_length
                        ).sum()
                    )

                elif (
                    column_action.aggregation.type
                    == AggregationType.SumProportionOfTarget
                ):
                    # data_to_aggregate_for_target_group_slk_length = data_to_aggregate_for_target_group[slk_to]-data_to_aggregate_for_target_group[slk_from]
                    target_length = target_row[slk_to] - target_row[slk_from]
                    aggregated_result_row.append(
                        (column_to_aggregate * column_to_aggregate_overlap_len).sum()
                        / target_length
                    )

                elif column_action.aggregation.type == AggregationType.Sum:
                    aggregated_result_row.append(column_to_aggregate.sum())

                elif column_action.aggregation.type == AggregationType.IndexOfMax:
                    aggregated_result_row.append(column_to_aggregate.idxmax())

                elif column_action.aggregation.type == AggregationType.IndexOfMin:
                    aggregated_result_row.append(column_to_aggregate.idxmin())

                elif column_action.aggregation.type == AggregationType.Max:
                    aggregated_result_row.append(column_to_aggregate.max())

                elif column_action.aggregation.type == AggregationType.Min:
                    aggregated_result_row.append(column_to_aggregate.min())

            result_index.append(target_index)
            result_rows.append(aggregated_result_row)

    result = target.join(
        pd.DataFrame(
            result_rows,
            columns=pd.Index([x.rename for x in column_actions]),
            index=pd.Index(result_index),
        )
    )
    if len(result.index) != len(target.index):
        raise Exception(
            "Oh no... the merge algorithim has somehow created addtional rows :O. This is a rare bug that I think is fixed now, but if you do see this message please contact the author."
        )
    duration = time.perf_counter() - start_time
    _emit_performance_event(
        "on_slk_intervals_legacy",
        duration=duration,
        groups=float(target_groups.ngroups),
        actions=float(len(column_actions)),
        rows=float(len(result_rows)),
    )
    return result


def _run_fast_merge(
    target: pd.DataFrame,
    data: pd.DataFrame,
    join_left: List[str],
    column_actions: List[Action],
    from_to: Tuple[str, str],
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Dispatch to the fastest available non-legacy implementation."""

    if not isinstance(target, pd.DataFrame) or not isinstance(data, pd.DataFrame):
        _validate_inputs(target, data, join_left, column_actions, from_to)

    categorical_fallback = _should_use_categorical_fallback(data, column_actions)
    numeric_index = is_numeric_dtype(data.index)
    if is_numba_available() and not categorical_fallback and numeric_index:
        return on_slk_intervals_numba(
            target=target,
            data=data,
            join_left=join_left,
            column_actions=column_actions,
            from_to=from_to,
            verbose=verbose,
        )

    return on_slk_intervals_optimized(
        target=target,
        data=data,
        join_left=join_left,
        column_actions=column_actions,
        from_to=from_to,
        verbose=verbose,
    )


def on_slk_intervals(
    target: pd.DataFrame,
    data: pd.DataFrame,
    join_left: List[str],
    column_actions: List[Action],
    from_to: Tuple[str, str],
    legacy: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Merge and aggregate interval data onto target segments.

    This is the main entry point for merging interval-based data. By default,
    it uses the fastest available optimized implementation. The ``legacy``
    flag is retained for backwards compatibility; set ``legacy=True`` to
    continue using the legacy implementation.
    When Numba is installed, that path uses the sparse Numba backend;
    otherwise it falls back to the vectorized dense-overlap implementation.

    Args:
        target: DataFrame containing the segments onto which data will be merged.
        data: DataFrame providing the measurements to aggregate.
        join_left: Ordered list of column names that define grouping keys.
        column_actions: Sequence of Action instances describing aggregations.
        from_to: Tuple containing the start and end column names for intervals.
        legacy: If False (default), uses the fastest available optimized
            implementation. If True, continues to use the legacy implementation
            for backwards compatibility.
        verbose: If True, prints diagnostic messages including fallback notices.
            If False (default), only shows progress bars and suppresses pandas warnings.

    Returns:
        A new DataFrame with the same rows as target plus aggregated columns.

    Examples:
        >>> # Use the fastest available optimized implementation (default)
        >>> result = on_slk_intervals(target, data, ["road"], actions, ("from", "to"))
        >>>
        >>> # Continue using the legacy implementation for backwards compatibility
        >>> result = on_slk_intervals(target, data, ["road"], actions, ("from", "to"), legacy=True)
        >>>
        >>> # Use optimized implementation with verbose output
        >>> result = on_slk_intervals(target, data, ["road"], actions, ("from", "to"), legacy=False, verbose=True)
    """
    # Keep the legacy flag for backwards compatibility; True selects the old path.
    if legacy:
        return on_slk_intervals_legacy(
            target=target,
            data=data,
            join_left=join_left,
            column_actions=column_actions,
            from_to=from_to,
            verbose=verbose,
        )

    return _run_fast_merge(
        target=target,
        data=data,
        join_left=join_left,
        column_actions=column_actions,
        from_to=from_to,
        verbose=verbose,
    )


def on_slk_intervals_optimized(
    target: pd.DataFrame,
    data: pd.DataFrame,
    join_left: List[str],
    column_actions: List[Action],
    from_to: Tuple[str, str],
    verbose: bool = False,
) -> pd.DataFrame:
    """Merge and aggregate interval data using the vectorised fast path.

    This implementation mirrors :func:`on_slk_intervals` but relies on
    NumPy-heavy vectorisation to reduce the number of per-row Python loops.
    During execution the function eagerly validates inputs, detects whether
    categorical fallbacks are required, and otherwise evaluates all
    aggregations for each target group using dense overlap matrices.

    Args:
        target: DataFrame containing the segments onto which data will be
            merged. Must contain the columns listed in ``join_left`` and the
            ``from_to`` bounds, have a unique, non-multi index, and avoid
            duplicate column names.
        data: DataFrame providing the measurements to aggregate. It must
            satisfy the same structural requirements as ``target`` and include
            any columns referenced by the supplied ``column_actions``.
        join_left: Ordered list of column names that define grouping keys for
            both ``target`` and ``data``. These keys are used to align rows
            before calculating overlap lengths.
        column_actions: Sequence of :class:`Action` instances describing which
            data columns to aggregate and how to label the outputs. The
            aggregation types are the same as those supported by the legacy
            merge path.
        from_to: Tuple containing the inclusive start column name and the
            exclusive end column name that describe each interval.

    Returns:
        A new DataFrame with the same rows (and index) as ``target`` plus one
        column per entry in ``column_actions`` containing the aggregated
        values.

    Raises:
        TypeError: If ``target`` or ``data`` are not DataFrames, or
        ``join_left`` is not a list.
        Exception: When structural validation fails (duplicate columns,
        missing join/interval columns, zero-length segments, or output column
        collisions).
        ValueError: When length-weighted percentiles lack a percentile value
        or when percentiles are outside the valid range.

    Notes:
        The function automatically defers to :func:`on_slk_intervals_fallback`
        when categorical aggregations (for ``KeepLongest`` on non-numeric
        columns) are detected. Both paths return identically structured
        outputs in order to remain drop-in replacements for the legacy merge
        helper.
    """
    start_time = time.perf_counter()
    slk_from, slk_to = from_to

    target, data = _validate_inputs(target, data, join_left, column_actions, from_to)

    # ---------- Pre-compute data groups for O(1) lookups ----------
    data_needed_cols = list({action.column_name for action in column_actions})
    data_subset = data.loc[:, [*join_left, slk_from, slk_to, *data_needed_cols]].copy()
    data_subset["_segment_len"] = data_subset[slk_to] - data_subset[slk_from]
    data_subset["_original_index"] = data_subset.index
    data_groups = _build_data_groups(data_subset, join_left)

    fallback_columns = _should_use_categorical_fallback(data_subset, column_actions)
    if fallback_columns:
        if verbose:
            print(
                f"[pyroads.merge] Falling back to categorical path for columns: {', '.join(fallback_columns)}"
            )
        result = on_slk_intervals_fallback(
            target=target,
            data=data,
            join_left=join_left,
            column_actions=column_actions,
            from_to=from_to,
            data_subset=data_subset,
            data_groups=data_groups,
            skip_validation=True,
            perf_origin="optimized",
            verbose=verbose,
        )
        duration = time.perf_counter() - start_time
        group_count = float(target.groupby(join_left, sort=False).ngroups)
        _emit_performance_event(
            "on_slk_intervals_optimized_delegated",
            duration=duration,
            actions=float(len(column_actions)),
            groups=group_count,
            fallback=float(len(fallback_columns)),
            rows=float(len(result.index)),
        )
        return result

    # ---------- Helper: reduce values according to an Action ----------
    def _aggregate(
        action: Action,
        values: np.ndarray,
        overlaps: np.ndarray,
        data_lengths: np.ndarray,
        target_length: float,
        source_indices: Optional[np.ndarray] = None,
    ):
        agg_type = action.aggregation.type

        if agg_type == AggregationType.Average:
            return values.mean()

        if agg_type == AggregationType.First:
            return values[0]

        if agg_type == AggregationType.LengthWeightedAverage:
            weights = overlaps
            return np.average(values, weights=weights)

        if agg_type == AggregationType.KeepLongestSegment:
            idx = overlaps.argmax()
            return values[idx]

        if agg_type == AggregationType.KeepLongest:
            return _keep_longest_tie_first(values, overlaps)

        if agg_type == AggregationType.LengthWeightedPercentile:
            percentile = action.aggregation.percentile
            if percentile is None:
                raise PercentileConfigurationError(
                    "LengthWeightedPercentile aggregation requires a percentile value."
                )
            order = np.argsort(values, kind="mergesort")
            sorted_vals = values[order]
            sorted_weights = overlaps[order].astype(float)
            if sorted_weights.size == 1:
                return float(sorted_vals[0])
            pair_avgs = (sorted_weights[:-1] + sorted_weights[1:]) * 0.5
            x_coords = np.zeros_like(sorted_weights)
            x_coords[1:] = np.cumsum(pair_avgs)
            total = x_coords[-1]
            if total <= 0:
                return float(sorted_vals[-1])
            coords = x_coords / total
            return np.interp(float(percentile), coords, sorted_vals)

        if agg_type == AggregationType.SumProportionOfData:
            # Sum(value * overlap / data_length)
            return np.sum(values * overlaps / data_lengths)

        if agg_type == AggregationType.SumProportionOfTarget:
            # Sum(value * overlap / target_length)
            return np.sum(values * overlaps) / target_length

        if agg_type == AggregationType.Sum:
            return values.sum()

        if agg_type == AggregationType.IndexOfMax:
            best_pos = int(values.argmax())
            if source_indices is not None and len(source_indices) > best_pos:
                return source_indices[best_pos]
            return best_pos
        if agg_type == AggregationType.IndexOfMin:
            best_pos = int(values.argmin())
            if source_indices is not None and len(source_indices) > best_pos:
                return source_indices[best_pos]
            return best_pos

        if agg_type == AggregationType.Max:
            return values.max()
        if agg_type == AggregationType.Min:
            return values.min()

        raise ValueError(f"Unsupported aggregation type: {agg_type}")

    # ---------- Main loop over target groups ----------
    result_buffer: list[list[Any]] = []
    result_index: list[Any] = []

    target_groups = target.groupby(join_left, sort=False)
    if verbose:
        print(
            f"[pyroads.merge] Using optimized path across {len(column_actions)} action(s) and {target_groups.ngroups} group(s)."
        )

    # Suppress pandas warnings unless verbose mode is enabled
    import warnings

    with warnings.catch_warnings():
        if not verbose:
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        for key, target_group in tqdm(
            target_groups, total=target_groups.ngroups, desc="pyroads.merge optimized"
        ):
            key_tuple = _normalize_group_key(key)
            data_group = data_groups.get(key_tuple)
            if data_group is None:
                continue  # no overlapping data → all NaN for these rows

            tgt_starts = target_group[slk_from].to_numpy(dtype=float)
            tgt_ends = target_group[slk_to].to_numpy(dtype=float)
            tgt_lengths = tgt_ends - tgt_starts

            data_starts = data_group[slk_from].to_numpy(dtype=float)
            data_ends = data_group[slk_to].to_numpy(dtype=float)
            data_lengths = data_group["_segment_len"].to_numpy(dtype=float)

            # overlap matrix: (n_target, n_data)
            overlap = np.minimum(tgt_ends[:, None], data_ends[None, :]) - np.maximum(
                tgt_starts[:, None], data_starts[None, :]
            )
            np.maximum(overlap, 0.0, out=overlap)

            if not np.any(overlap):
                # No overlaps at all for this group; skip (all NaNs for each row)
                continue

            # Pre-extract action columns as NumPy arrays for fast slicing
            action_arrays = {
                action.column_name: data_group[action.column_name].to_numpy()
                for action in column_actions
            }
            original_indices_arr = data_group["_original_index"].to_numpy()

            for local_idx, tgt_idx in enumerate(target_group.index):
                overlap_row = overlap[local_idx]
                mask = overlap_row > 0
                if not mask.any():
                    continue  # this target row gets NaNs (handled after loop)
                target_length = tgt_lengths[local_idx]

                aggregated_values: list[Any] = []
                for action in column_actions:
                    values = action_arrays[action.column_name][mask]
                    overlaps = overlap_row[mask]
                    lengths = data_lengths[mask]
                    indices = original_indices_arr[mask]
                    if len(values) == 0:
                        aggregated_values.append(np.nan)
                        continue

                    valid = ~pd.isna(values)
                    if not valid.any():
                        aggregated_values.append(np.nan)
                        continue

                    values = values[valid]
                    overlaps = overlaps[valid]
                    lengths = lengths[valid]
                    indices = indices[valid]

                    agg_value = _aggregate(
                        action,
                        values=values,
                        overlaps=overlaps,
                        data_lengths=lengths,
                        target_length=target_length,
                        source_indices=indices,
                    )
                    aggregated_values.append(agg_value)

                result_buffer.append(aggregated_values)
                result_index.append(tgt_idx)

    # ---------- Assemble output ----------
    merged = target.join(
        pd.DataFrame(
            result_buffer,
            index=pd.Index(result_index),
            columns=pd.Index([action.rename for action in column_actions]),
        )
    )

    if len(merged) != len(target):
        raise RuntimeError(
            "Unexpected row-count mismatch after merge; please report this case."
        )

    duration = time.perf_counter() - start_time
    _emit_performance_event(
        "on_slk_intervals_optimized",
        duration=duration,
        groups=float(target_groups.ngroups),
        actions=float(len(column_actions)),
        rows=float(len(result_index)),
        fallback=0.0,
    )

    return merged


def _detect_dataframe_backend(target: object, data: object) -> str:
    """Detect whether inputs are pandas, polars, or dask DataFrames.

    Returns "pandas", "polars", or "dask". Raises ``TypeError`` if `target`
    and `data` belong to different backends.
    """

    def _backend_of(frame: object) -> Optional[str]:
        if isinstance(frame, pd.DataFrame):
            return "pandas"
        try:
            import polars as pl

            if isinstance(frame, (pl.DataFrame, pl.LazyFrame)):
                return "polars"
        except ImportError:
            pass
        try:
            import dask.dataframe as dd

            if isinstance(frame, dd.DataFrame):
                return "dask"
        except ImportError:
            pass
        return None

    target_backend = _backend_of(target)
    data_backend = _backend_of(data)

    if target_backend is None or data_backend is None:
        # Fall through to pandas validation, which will raise a clear,
        # consistent InvalidDataFrameError for unsupported types.
        return "pandas"

    if target_backend != data_backend:
        raise TypeError(
            "`target` and `data` must be the same DataFrame type. Got "
            f"target={target_backend!r} ({type(target)}) and "
            f"data={data_backend!r} ({type(data)})."
        )

    return target_backend


def on_slk_intervals_auto(
    target: Any,
    data: Any,
    join_left: List[str],
    column_actions: List[Action],
    from_to: Tuple[str, str],
    prefer_optimized: Optional[bool] = None,
) -> Any:
    """Dispatch to the appropriate merge backend and implementation.

    In addition to pandas DataFrames, `target` and `data` may both be Polars
    DataFrames or both be Dask DataFrames -- the input type is detected
    automatically and routed to the matching backend
    (:func:`~pyroads.merge._polars_merge.on_slk_intervals_polars` or
    :func:`~pyroads.merge._dask_merge.on_slk_intervals_dask`). Mixing
    backends between `target` and `data` raises ``TypeError``.

    Args:
        target, data, join_left, column_actions, from_to: See
            :func:`on_slk_intervals` for parameter descriptions. `target` and
            `data` may be pandas, Polars, or Dask DataFrames (both must be the
            same type).
        prefer_optimized: When ``True`` the fastest available optimized path is
            used, when ``False`` the legacy implementation is enforced. If
            ``None`` (default), the behaviour is controlled by the
            ``MERGE_SEGMENTS_DEFAULT_MODE`` environment variable
            (``"optimized"`` or ``"legacy"``). When the variable is unset the
            optimized implementation is preferred. This parameter only applies
            to the pandas backend; Polars and Dask always use their optimized
            (Numba-backed) paths.

    Returns:
        The merged DataFrame, in the same type as the input (pandas, Polars,
        or a lazy Dask DataFrame).
    """

    backend = _detect_dataframe_backend(target, data)

    if backend == "polars":
        from ._polars_merge import on_slk_intervals_polars

        return on_slk_intervals_polars(
            target=target,
            data=data,
            join_left=join_left,
            column_actions=column_actions,
            from_to=from_to,
        )

    if backend == "dask":
        from ._dask_merge import on_slk_intervals_dask

        return on_slk_intervals_dask(
            target=target,
            data=data,
            join_left=join_left,
            column_actions=column_actions,
            from_to=from_to,
        )

    if prefer_optimized is None:
        mode = os.getenv("MERGE_SEGMENTS_DEFAULT_MODE")
        if mode:
            mode_lower = mode.lower()
            if mode_lower in {"optimized", "fast", "vectorized"}:
                prefer_optimized = True
            elif mode_lower in {"legacy", "safe", "fallback"}:
                prefer_optimized = False

    if prefer_optimized is None:
        prefer_optimized = True

    if prefer_optimized:
        return _run_fast_merge(
            target=target,
            data=data,
            join_left=join_left,
            column_actions=column_actions,
            from_to=from_to,
        )

    return on_slk_intervals_legacy(
        target=target,
        data=data,
        join_left=join_left,
        column_actions=column_actions,
        from_to=from_to,
    )


def on_slk_intervals_fallback(
    target: pd.DataFrame,
    data: pd.DataFrame,
    join_left: List[str],
    column_actions: List[Action],
    from_to: Tuple[str, str],
    *,
    data_subset: Any = None,
    data_groups: Optional[Dict[tuple, pd.DataFrame]] = None,
    skip_validation: bool = False,
    perf_origin: str = "legacy",
    verbose: bool = False,
) -> pd.DataFrame:
    """Merge interval data using the categorical-friendly fallback path.

    The fallback logic is intentionally close to the original implementation
    in :func:`on_slk_intervals`. It exists to handle scenarios where
    categorical aggregations (for example ``KeepLongest`` on string columns)
    require stable ordering guarantees that the vectorised fast path cannot
    provide. When invoked directly it performs the same validation as the
    optimized function before iterating through each target row and computing
    overlaps in pure Python.

    Args:
        target: DataFrame containing the intervals to populate. It must have
            the grouping columns listed in ``join_left`` as well as the start
            and end columns provided in ``from_to``.
        data: DataFrame providing the measurement values to aggregate. All
            referenced join, interval, and aggregation columns must be
            present, and indices/column labels must be unique and non-multi.
        join_left: Ordered list of column names identifying grouping keys used
            to align ``target`` and ``data`` prior to calculating overlaps.
        column_actions: Aggregations to apply for each data column. Each
            :class:`Action` specifies the source column, aggregation strategy,
            and output column name.
        from_to: Tuple naming the start and end columns describing each
            interval. The function assumes the values form half-open ranges
            (start inclusive, end exclusive) and rejects zero-length rows.
        data_subset: Optional pre-filtered ``data`` slice containing only the
            join columns, bounds, and required aggregation inputs. Supplying
            this avoids redundant copying when the fallback is invoked from
            :func:`on_slk_intervals_optimized`.
        data_groups: Optional dictionary mapping normalized join keys to
            grouped sub-dataframes. As with ``data_subset``, this parameter is
            primarily used by the optimized wrapper to reuse pre-computed
            structures after detecting categorical columns.

    Returns:
        DataFrame equal in length (and index) to ``target`` with extra columns
        produced by ``column_actions``.

    Raises:
        TypeError: If ``target`` or ``data`` are not DataFrames.
        Exception: When structural validation fails or when zero-length
        segments are encountered.
        ValueError: For invalid percentile arguments supplied to
        ``LengthWeightedPercentile`` aggregations.

    Notes:
        Although slower than :func:`on_slk_intervals_optimized`, the fallback
        path guarantees consistent ordering for categorical aggregations and
        therefore serves as a correctness-preserving safety net when the input
        data contains non-numeric values.
    """
    start_time = time.perf_counter()
    slk_from, slk_to = from_to

    if skip_validation:
        target_df = target
        data_df = data
    else:
        target_df, data_df = _validate_inputs(
            target, data, join_left, column_actions, from_to
        )
    target = target_df
    data = data_df

    if data_subset is None:
        data_needed_cols = list({action.column_name for action in column_actions})
        data_subset = data.loc[
            :, [*join_left, slk_from, slk_to, *data_needed_cols]
        ].copy()
        data_subset["_segment_len"] = data_subset[slk_to] - data_subset[slk_from]
        data_subset["_original_index"] = data_subset.index

    if data_groups is None:
        data_groups = _build_data_groups(data_subset, join_left)

    target_groups = target.groupby(join_left, sort=False)
    if verbose:
        print(
            f"[pyroads.merge] Running categorical fallback across {len(column_actions)} action(s) and {target_groups.ngroups} group(s)."
        )

    def _aggregate(
        action: Action,
        values: np.ndarray,
        overlaps: np.ndarray,
        data_lengths: np.ndarray,
        target_length: float,
        source_indices: Optional[np.ndarray] = None,
    ):
        agg_type = action.aggregation.type

        if agg_type == AggregationType.Average:
            return values.mean()

        if agg_type == AggregationType.First:
            return values[0]

        if agg_type == AggregationType.LengthWeightedAverage:
            weights = overlaps
            return np.average(values, weights=weights)

        if agg_type == AggregationType.KeepLongestSegment:
            idx_max = overlaps.argmax()
            return values[idx_max]

        if agg_type == AggregationType.KeepLongest:
            return _keep_longest_tie_first(values, overlaps)

        if agg_type == AggregationType.LengthWeightedPercentile:
            percentile = action.aggregation.percentile
            if percentile is None:
                raise PercentileConfigurationError(
                    "LengthWeightedPercentile aggregation requires a percentile value."
                )
            order = np.argsort(values, kind="mergesort")
            sorted_vals = values[order]
            sorted_weights = overlaps[order].astype(float)
            if sorted_weights.size == 1:
                return float(sorted_vals[0])
            pair_avgs = (sorted_weights[:-1] + sorted_weights[1:]) * 0.5
            x_coords = np.zeros_like(sorted_weights)
            x_coords[1:] = np.cumsum(pair_avgs)
            total = x_coords[-1]
            if total <= 0:
                return float(sorted_vals[-1])
            coords = x_coords / total
            return np.interp(float(percentile), coords, sorted_vals)

        if agg_type == AggregationType.SumProportionOfData:
            return np.sum(values * overlaps / data_lengths)

        if agg_type == AggregationType.SumProportionOfTarget:
            return np.sum(values * overlaps) / target_length

        if agg_type == AggregationType.Sum:
            return values.sum()

        if agg_type == AggregationType.IndexOfMax:
            best_pos = int(values.argmax())
            if source_indices is not None and len(source_indices) > best_pos:
                return source_indices[best_pos]
            return best_pos
        if agg_type == AggregationType.IndexOfMin:
            best_pos = int(values.argmin())
            if source_indices is not None and len(source_indices) > best_pos:
                return source_indices[best_pos]
            return best_pos

        if agg_type == AggregationType.Max:
            return values.max()
        if agg_type == AggregationType.Min:
            return values.min()

        raise ValueError(f"Unsupported aggregation type: {agg_type}")

    result_buffer: list[list[Any]] = []
    result_index: list[Any] = []

    # Suppress pandas warnings unless verbose mode is enabled
    import warnings

    with warnings.catch_warnings():
        if not verbose:
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        for key, target_group in tqdm(
            target_groups, total=target_groups.ngroups, desc="pyroads.merge fallback"
        ):
            key_tuple = _normalize_group_key(key)
            data_group = data_groups.get(key_tuple)
            if data_group is None:
                continue

            tgt_starts = target_group[slk_from].to_numpy(dtype=float)
            tgt_ends = target_group[slk_to].to_numpy(dtype=float)
            tgt_lengths = tgt_ends - tgt_starts

            data_starts = data_group[slk_from].to_numpy(dtype=float)
            data_ends = data_group[slk_to].to_numpy(dtype=float)
            data_lengths = data_group["_segment_len"].to_numpy(dtype=float)

            overlap = np.minimum(tgt_ends[:, None], data_ends[None, :]) - np.maximum(
                tgt_starts[:, None], data_starts[None, :]
            )
            np.maximum(overlap, 0.0, out=overlap)

            if not np.any(overlap):
                continue

            action_arrays = {
                action.column_name: data_group[action.column_name].to_numpy()
                for action in column_actions
            }
            original_indices_arr = data_group["_original_index"].to_numpy()

            for local_idx, tgt_idx in enumerate(target_group.index):
                overlap_row = overlap[local_idx]
                mask = overlap_row > 0
                if not mask.any():
                    continue
                target_length = tgt_lengths[local_idx]

                aggregated_values: list[Any] = []
                for action in column_actions:
                    values = action_arrays[action.column_name][mask]
                    overlaps = overlap_row[mask]
                    lengths = data_lengths[mask]
                    indices = original_indices_arr[mask]
                    if len(values) == 0:
                        aggregated_values.append(np.nan)
                        continue

                    valid = ~pd.isna(values)
                    if not valid.any():
                        aggregated_values.append(np.nan)
                        continue

                    values = values[valid]
                    overlaps = overlaps[valid]
                    lengths = lengths[valid]
                    indices = indices[valid]

                    agg_value = _aggregate(
                        action,
                        values=values,
                        overlaps=overlaps,
                        data_lengths=lengths,
                        target_length=target_length,
                        source_indices=indices,
                    )
                    aggregated_values.append(agg_value)

                result_buffer.append(aggregated_values)
                result_index.append(tgt_idx)

    merged = target.join(
        pd.DataFrame(
            result_buffer,
            index=pd.Index(result_index),
            columns=pd.Index([action.rename for action in column_actions]),
        )
    )

    if len(merged) != len(target):
        raise RuntimeError(
            "Unexpected row-count mismatch after merge; please report this case."
        )

    duration = time.perf_counter() - start_time
    event_name = (
        "on_slk_intervals_fallback_from_optimized"
        if perf_origin == "optimized"
        else "on_slk_intervals_fallback"
    )
    _emit_performance_event(
        event_name,
        duration=duration,
        groups=float(target_groups.ngroups),
        actions=float(len(column_actions)),
        rows=float(len(result_buffer)),
    )

    return merged


_configure_default_performance_logger_from_env()
