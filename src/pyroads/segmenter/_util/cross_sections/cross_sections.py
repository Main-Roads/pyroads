from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd

from ..check_segmentation import check_linear_index, check_linear_index_is_ordered_and_disjoint
from .._kernels import _cross_sections_numba


class CN:
    group_number = "__group_number__"
    event_measure_true = "__event_measure_true__"
    event_measure_slk = "__event_measure_slk__"
    event_type = "__event_type__"
    event_measure_diff = "__event_measure_diff__"


Snapshot = tuple[tuple[Hashable, ...], tuple[tuple[Hashable, float], ...]]
Transition = tuple[float, float, float, float, tuple[Snapshot, ...]]


def _active_cross_section_snapshots(
    events: pd.DataFrame,
    cross_section_categories: list[str],
) -> list[tuple[Snapshot, ...]]:
    active: dict[tuple[Hashable, ...], dict[Hashable, int]] = {}
    snapshots: list[tuple[Snapshot, ...]] = []
    event_columns = [*cross_section_categories, CN.event_type]

    for row in events[event_columns].itertuples(index=True, name=None):
        event_index = row[0]
        category_path = tuple(row[1:1 + len(cross_section_categories)])
        path_data = active.setdefault(category_path, {})
        if row[-1] == "start":
            path_data[event_index] = 1
        else:
            path_data.pop(event_index, None)
            if not path_data:
                del active[category_path]
        snapshots.append(tuple(
            (path, tuple(path_data.items()))
            for path, path_data in active.items()
        ))
    return snapshots


def _merge_snapshot_data(left: tuple[Snapshot, ...], right: tuple[Snapshot, ...]) -> tuple[Snapshot, ...]:
    merged = []
    for (left_path, left_data), (right_path, right_data) in zip(left, right):
        if left_path != right_path:
            raise ValueError("Cannot merge cross-section snapshots with different paths")
        data_by_index: dict[Hashable, float] = {}
        for index, value in (*left_data, *right_data):
            data_by_index[index] = data_by_index.get(index, 0) + value
        merged.append((left_path, tuple(sorted(data_by_index.items()))))
    return tuple(merged)


def _merge_cross_section_transitions(
    event_rows: list[tuple[float, float, float, tuple[Snapshot, ...]]],
) -> list[Transition]:
    transitions: list[Transition] = []
    for row_index in range(1, len(event_rows)):
        previous = event_rows[row_index - 1]
        current = event_rows[row_index]
        if current[2] <= 0:
            continue
        transition_length = current[0] - previous[0]
        weighted_snapshot = tuple(
            (path, tuple((index, transition_length * value) for index, value in path_data))
            for path, path_data in previous[3]
        )
        transition: Transition = (previous[0], current[0], previous[1], current[1], weighted_snapshot)
        previous_paths = tuple(item[0] for item in transitions[-1][4]) if transitions else ()
        current_paths = tuple(item[0] for item in transition[4])
        if transitions and previous_paths == current_paths:
            previous_transition = transitions[-1]
            transitions[-1] = (
                previous_transition[0],
                transition[1],
                previous_transition[2],
                transition[3],
                _merge_snapshot_data(previous_transition[4], transition[4]),
            )
        else:
            transitions.append(transition)
    return transitions


def cross_sections(
    segmentation: Any,
    group_categories: list[str],
    cross_section_categories: list[str],
    measure_slk: tuple[str, str],
    measure_true: tuple[str, str],
    out_col_name_cross_section_number: str = "cross_section_number",
    out_col_name_original_index: str = "original_index",
    out_col_name_overlap: str = "overlap",
) -> pd.DataFrame:
    """Return intervals grouped by their active cross-section categories.

    The input must be disjoint within each group and cross-section-category
    combination. Returned overlaps are measured in the true measure.
    """
    check_linear_index(segmentation[list(measure_slk)])
    check_linear_index(segmentation[list(measure_true)])
    check_linear_index_is_ordered_and_disjoint(
        segmentation, measure_true, [*group_categories, *cross_section_categories]
    )

    output_rows = []
    group_by = group_categories[0] if len(group_categories) == 1 else group_categories
    for group_counter, (group_index, group) in enumerate(segmentation.groupby(group_by)):
        group = group[[*group_categories, *cross_section_categories, *measure_true, *measure_slk]]
        start_events = group.copy().sort_values(by=cross_section_categories, ascending=True)
        start_events[CN.event_measure_true] = start_events[measure_true[0]]
        start_events[CN.event_measure_slk] = start_events[measure_slk[0]]
        start_events[CN.event_type] = "start"
        end_events = group.copy().sort_values(by=cross_section_categories, ascending=False)
        end_events[CN.event_measure_true] = end_events[measure_true[1]]
        end_events[CN.event_measure_slk] = end_events[measure_slk[1]]
        end_events[CN.event_type] = "end"
        events = pd.concat([end_events, start_events], axis="index").sort_values(
            by=CN.event_measure_true, kind="stable"
        )
        events[CN.event_measure_diff] = events[CN.event_measure_true] - events[CN.event_measure_true].shift(
            1, fill_value=events[CN.event_measure_true].iloc[0]
        )
        if _cross_sections_numba is not None:
            path_ids: dict[tuple[Hashable, ...], int] = {}
            path_values: list[tuple[Hashable, ...]] = []
            event_path = []
            for category_path in events[cross_section_categories].itertuples(index=False, name=None):
                category_path = tuple(category_path)
                path_id = path_ids.setdefault(category_path, len(path_ids))
                if path_id == len(path_values):
                    path_values.append(category_path)
                event_path.append(path_id)

            source_ids: dict[Hashable, int] = {}
            source_labels: list[Hashable] = []
            for event_index, event_type in zip(events.index, events[CN.event_type]):
                if event_type == "start" and event_index not in source_ids:
                    source_ids[event_index] = len(source_labels)
                    source_labels.append(event_index)
            event_source = np.asarray([source_ids[event_index] for event_index in events.index], dtype=np.int64)
            event_measure_true = events[CN.event_measure_true].to_numpy(dtype=np.float64)
            event_measure_slk = events[CN.event_measure_slk].to_numpy(dtype=np.float64)
            event_type = (events[CN.event_type].to_numpy() == "end").astype(np.int64)
            event_path_array = np.asarray(event_path, dtype=np.int64)
            empty_outputs: list[Any] = [np.empty(0, dtype=np.float64) for _ in range(4)]
            empty_outputs.extend([np.empty(0, dtype=np.int64) for _ in range(2)])
            empty_outputs.append(np.empty(0, dtype=np.float64))
            empty_outputs.append(np.empty(0, dtype=np.int64))
            output_count = _cross_sections_numba(
                event_measure_true,
                event_measure_slk,
                event_type,
                event_path_array,
                event_source,
                *empty_outputs,
            )
            outputs: list[Any] = [np.empty(output_count, dtype=np.float64) for _ in range(4)]
            outputs.extend([np.empty(output_count, dtype=np.int64) for _ in range(2)])
            outputs.append(np.empty(output_count, dtype=np.float64))
            outputs.append(np.empty(output_count, dtype=np.int64))
            _cross_sections_numba(
                event_measure_true,
                event_measure_slk,
                event_type,
                event_path_array,
                event_source,
                *outputs,
            )
            group_index_list = [group_index] if not isinstance(group_index, tuple) else group_index
            for row_index in range(output_count):
                path = path_values[outputs[4][row_index]]
                output_rows.append([
                    group_counter,
                    outputs[7][row_index],
                    *group_index_list,
                    *path,
                    outputs[0][row_index],
                    outputs[1][row_index],
                    outputs[2][row_index],
                    outputs[3][row_index],
                    source_labels[outputs[5][row_index]],
                    outputs[6][row_index],
                ])
            continue
        snapshots = _active_cross_section_snapshots(events, cross_section_categories)
        event_rows = [
            (*row, snapshot)
            for row, snapshot in zip(
                events[[CN.event_measure_true, CN.event_measure_slk, CN.event_measure_diff]].itertuples(
                    index=False, name=None
                ),
                snapshots,
            )
        ]
        merged_transitions = _merge_cross_section_transitions(event_rows)
        group_index_list = [group_index] if not isinstance(group_index, tuple) else group_index
        for cross_section_number, transition in enumerate(merged_transitions):
            true_from, true_to, slk_from, slk_to, snapshot = transition
            for child_name, child_data in snapshot:
                for original_index, overlap in child_data:
                    output_rows.append([
                        group_counter,
                        cross_section_number,
                        *group_index_list,
                        *child_name,
                        true_from,
                        true_to,
                        slk_from,
                        slk_to,
                        original_index,
                        overlap,
                    ])

    result: Any = pd.DataFrame(
        output_rows,
        columns=pd.Index([
            CN.group_number,
            out_col_name_cross_section_number,
            *group_categories,
            *cross_section_categories,
            *measure_true,
            *measure_slk,
            out_col_name_original_index,
            out_col_name_overlap,
        ]),
    )
    result.loc[:, out_col_name_cross_section_number] = (
        result.loc[:, out_col_name_cross_section_number]
        + result[[CN.group_number]].join(
            (result.groupby(CN.group_number)[out_col_name_cross_section_number].max() + 1)
            .cumsum()
            .shift(1, fill_value=0),
            on=CN.group_number,
        ).loc[:, out_col_name_cross_section_number]
    )
    return result.drop(columns=[CN.group_number])


def cross_sections_normalised(
    segmentation: pd.DataFrame,
    group_categories: list[str],
    cross_section_categories: list[str],
    measure_slk: tuple[str, str],
    measure_true: tuple[str, str],
    out_col_name_cross_section_number: str = "cross_section_number",
    out_col_name_original_index: str = "original_index",
    out_col_name_overlap: str = "overlap",
):
    """Return a group table and normalized cross-section table."""
    result = cross_sections(
        segmentation=segmentation,
        group_categories=group_categories,
        cross_section_categories=cross_section_categories,
        measure_slk=measure_slk,
        measure_true=measure_true,
        out_col_name_cross_section_number=out_col_name_cross_section_number,
        out_col_name_original_index=out_col_name_original_index,
        out_col_name_overlap=out_col_name_overlap,
    )
    group_table = result[[
        out_col_name_cross_section_number,
        *group_categories,
        *measure_true,
        *measure_slk,
    ]].drop_duplicates().reset_index(drop=True)
    cross_section_table = result[[
        out_col_name_cross_section_number,
        *cross_section_categories,
        out_col_name_original_index,
        out_col_name_overlap,
    ]]
    return group_table, cross_section_table
