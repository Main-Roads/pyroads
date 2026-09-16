"""Small adapters shared by pandas- and Polars-facing public APIs."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import pandas as pd


def _polars_module() -> Any:
    try:
        import polars as pl
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "polars is required when a Polars DataFrame is passed. "
            "Install with: pip install pyroads[polars]"
        ) from error
    return pl


def _is_polars_dataframe(value: Any) -> bool:
    try:
        pl = _polars_module()
    except ImportError:
        return False
    return isinstance(value, pl.DataFrame)


def _to_pandas(value: Any) -> Any:
    if _is_polars_dataframe(value):
        return pd.DataFrame(value.to_dict(as_series=False))
    if isinstance(value, tuple):
        return tuple(_to_pandas(item) for item in value)
    if isinstance(value, list):
        return [_to_pandas(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_pandas(item) for key, item in value.items()}
    return value


def _to_polars(value: Any, pl: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        frame = value.reset_index() if isinstance(value.index, pd.MultiIndex) else value
        return pl.DataFrame(frame.to_dict(orient="list"))
    if isinstance(value, pd.Series):
        return pl.Series(value.tolist())
    if isinstance(value, tuple):
        return tuple(_to_polars(item, pl) for item in value)
    if isinstance(value, list):
        return [_to_polars(item, pl) for item in value]
    if isinstance(value, dict):
        return {key: _to_polars(item, pl) for key, item in value.items()}
    return value


def supports_pandas_and_polars(function: Callable[..., Any]) -> Callable[..., Any]:
    """Allow a public pandas implementation to accept eager Polars frames."""

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        values = (*args, *kwargs.values())
        uses_polars = any(_is_polars_dataframe(value) for value in values)
        if not uses_polars:
            return function(*args, **kwargs)

        pl = _polars_module()
        result = function(
            *(_to_pandas(value) for value in args),
            **{key: _to_pandas(value) for key, value in kwargs.items()},
        )
        return _to_polars(result, pl)

    return wrapper