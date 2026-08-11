from __future__ import annotations

from dataclasses import dataclass

from ..merge import Aggregation, Action as IntervalAction, on_slk_intervals

MODE = "mode"
SUM = "sum"


@dataclass
class Action:
    column_name: str
    rename: str | None = None
    aggregation: str = MODE


def on_intervals(
    left_df,
    right_df,
    idvars,
    start,
    end,
    column_actions,
):
    """Compatibility adapter for the historical interval-merge interface."""
    actions = []
    for action in column_actions:
        aggregation = (
            Aggregation.Sum()
            if action.aggregation == SUM
            else Aggregation.KeepLongest()
        )
        actions.append(
            IntervalAction(
                action.column_name,
                aggregation,
                rename=action.rename or action.column_name,
            )
        )
    return on_slk_intervals(
        target=left_df,
        data=right_df,
        join_left=idvars,
        column_actions=actions,
        from_to=(start, end),
    )


__all__ = ["Action", "MODE", "SUM", "on_intervals"]
