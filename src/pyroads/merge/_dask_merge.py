"""Dask-backed interval merge implementation.

``target`` and ``data`` are repartitioned so that rows sharing the same
``join_left`` key co-locate in matching partitions (a hash-based shuffle),
then each partition pair is delegated to the existing pandas/Numba merge
implementation via ``dask.dataframe.map_partitions``. Crucially, this module
never calls ``.compute()`` internally: the returned object is a **lazy**
``dask.dataframe.DataFrame`` graph. The caller decides when (and how) to
materialize it -- e.g. ``.compute()`` for an in-memory pandas result, or
``.to_parquet(...)`` to stream results to disk without ever holding the full
dataset in memory. This is what allows datasets larger than RAM to be merged:
only one partition's worth of data is materialized per task.

Requires: dask[dataframe].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .exceptions import InvalidDataFrameError

if TYPE_CHECKING:
    import dask.dataframe as dd

    DASK_AVAILABLE = True
else:
    try:
        import dask.dataframe as dd

        DASK_AVAILABLE = True
    except ImportError:  # pragma: no cover - optional dependency
        dd = None
        DASK_AVAILABLE = False


def is_dask_available() -> bool:
    """Check if Dask is available for the Dask-native merge backend."""
    return DASK_AVAILABLE


def _ensure_dask_dataframe(name: str, frame: object) -> dd.DataFrame:
    if dd is None:
        raise ImportError(
            "dask is required for on_slk_intervals_dask. "
            "Install with: pip install pyroads[dask]"
        )
    if not isinstance(frame, dd.DataFrame):
        raise InvalidDataFrameError(
            f"`{name}` parameter must be a dask DataFrame, received {type(frame)}"
        )
    return frame


def _validate_dask_inputs(
    target: object,
    data: object,
    join_left: List[str],
    from_to: Tuple[str, str],
) -> Tuple[dd.DataFrame, dd.DataFrame]:
    if not isinstance(join_left, list):
        raise TypeError("`join_left` must be a list of column names.")

    target_df = _ensure_dask_dataframe("target", target)
    data_df = _ensure_dask_dataframe("data", data)

    slk_from, slk_to = from_to
    required = [*join_left, slk_from, slk_to]
    missing_messages: List[str] = []
    for column_name in required:
        in_target = column_name in target_df.columns
        in_data = column_name in data_df.columns
        if not in_target and not in_data:
            missing_messages.append(
                f"Column '{column_name}' is missing from both `target` and `data`."
            )
        elif not in_target:
            missing_messages.append(f"Column '{column_name}' is missing from `target`.")
        elif not in_data:
            missing_messages.append(f"Column '{column_name}' is missing from `data`.")
    if missing_messages:
        from .exceptions import InvalidJoinConfigurationError

        raise InvalidJoinConfigurationError(
            "Please check the `join_left` and `from_to` parameters. "
            "Specified columns must be present and have matching names in both "
            "`target` and `data`:\n" + "\n".join(missing_messages)
        )

    return target_df, data_df


def _build_meta(
    target: dd.DataFrame,
    data: dd.DataFrame,
    column_actions: List[Any],
) -> pd.DataFrame:
    """Build an empty pandas DataFrame describing the output schema for Dask."""
    meta = target._meta.copy()
    data_meta = data._meta
    for action in column_actions:
        aggregation_name = action.aggregation.type.name
        column_dtype = (
            data_meta[action.column_name].dtype
            if action.column_name in data_meta.columns
            else np.dtype("float64")
        )
        if aggregation_name == "KeepLongest" and not pd.api.types.is_numeric_dtype(
            column_dtype
        ):
            meta[action.rename] = pd.Series([], dtype=object)
        else:
            meta[action.rename] = pd.Series([], dtype="float64")
    return meta


def on_slk_intervals_dask(
    target: dd.DataFrame,
    data: dd.DataFrame,
    join_left: List[str],
    column_actions: List[Any],
    from_to: Tuple[str, str],
    verbose: bool = False,
    npartitions: Optional[int] = None,
) -> dd.DataFrame:
    """Merge and aggregate interval data using an out-of-core Dask path.

    ``target`` and ``data`` are shuffled so that rows sharing the same
    ``join_left`` key land in matching partitions, and each partition pair is
    merged independently using the existing pandas/Numba implementation. No
    ``.compute()`` is called internally -- the returned Dask DataFrame is
    still lazy, so it can be materialized incrementally (e.g. streamed to
    Parquet) without ever loading the whole dataset into memory at once.

    Args:
        target: Dask DataFrame containing the segments to populate.
        data: Dask DataFrame providing the measurements to aggregate.
        join_left: Ordered list of column names defining grouping keys.
        column_actions: Sequence of :class:`Action` instances describing
            aggregations.
        from_to: Tuple of (start column, end column) names describing each
            interval.
        verbose: If True, prints a message describing the constructed graph.
        npartitions: Number of partitions to shuffle both frames into.
            Defaults to ``max(data.npartitions, target.npartitions)``.

    Returns:
        A lazy Dask DataFrame. Call ``.compute()`` for an in-memory pandas
        result, or write it out incrementally (e.g. ``.to_parquet(...)``) to
        avoid materializing the full result at once.

    Note:
        This requires that no single ``join_left`` group is larger than what
        fits in memory for one partition -- the underlying algorithm still
        needs a full group's data in memory to compute overlaps for it.
    """
    target_df, data_df = _validate_dask_inputs(target, data, join_left, from_to)

    n_partitions = npartitions or max(data_df.npartitions, target_df.npartitions, 1)

    target_shuffled = target_df.shuffle(on=join_left, npartitions=n_partitions)
    data_shuffled = data_df.shuffle(on=join_left, npartitions=n_partitions)

    meta = _build_meta(target_shuffled, data_shuffled, column_actions)

    # Lazy import: avoids a module-level circular import with `merge.py`,
    # and ensures we reuse the exact same pandas/Numba dispatch logic.
    from .merge import on_slk_intervals_auto

    def _merge_partition(
        target_partition: pd.DataFrame, data_partition: pd.DataFrame
    ) -> pd.DataFrame:
        if target_partition.empty:
            return meta.head(0).copy()
        if data_partition.empty:
            result = target_partition.copy()
            for action in column_actions:
                if meta[action.rename].dtype == object:
                    result[action.rename] = None
                else:
                    result[action.rename] = np.nan
            return result
        return on_slk_intervals_auto(
            target=target_partition,
            data=data_partition,
            join_left=join_left,
            column_actions=column_actions,
            from_to=from_to,
            prefer_optimized=True,
        )

    result = dd.map_partitions(
        _merge_partition,
        target_shuffled,
        data_shuffled,
        meta=meta,
    )

    if verbose:
        print(
            f"[pyroads.merge] Dask merge graph built across {n_partitions} "
            "partition(s). Call `.compute()` (or `.to_parquet(...)`) to execute."
        )

    return result
