from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from pyroads._backend import announce_fallback

try:
    import pyroads._native as _rust_native
except ImportError:
    _rust_native = None

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised when performance extras are absent
    njit: Any = None


if njit is not None:

    @njit(cache=True, nogil=True)
    def _longest_overlap_positions_numba(
        output_from: npt.NDArray[np.float64],
        output_to: npt.NDArray[np.float64],
        source_from: npt.NDArray[np.float64],
        source_to: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.int64]:
        positions = np.full(output_from.size, -1, dtype=np.int64)
        source_start = 0

        for output_index in range(output_from.size):
            while source_start < source_from.size and source_to[source_start] <= output_from[output_index]:
                source_start += 1

            candidate = source_start
            best_overlap = 0.0
            best_position = -1
            while candidate < source_from.size and source_from[candidate] < output_to[output_index]:
                overlap = min(output_to[output_index], source_to[candidate]) - max(
                    output_from[output_index], source_from[candidate]
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_position = candidate
                candidate += 1

            positions[output_index] = best_position

        return positions

else:
    _longest_overlap_positions_numba = None


def longest_overlap_positions(
    output_from: npt.NDArray[np.float64],
    output_to: npt.NDArray[np.float64],
    source_from: npt.NDArray[np.float64],
    source_to: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """Select the first source interval with the greatest positive overlap."""
    if _rust_native is not None:
        return _rust_native.longest_overlap_positions(
            np.ascontiguousarray(output_from, dtype=np.float64),
            np.ascontiguousarray(output_to, dtype=np.float64),
            np.ascontiguousarray(source_from, dtype=np.float64),
            np.ascontiguousarray(source_to, dtype=np.float64),
        )
    announce_fallback()
    if _longest_overlap_positions_numba is not None:
        return _longest_overlap_positions_numba(output_from, output_to, source_from, source_to)

    positions = np.full(output_from.size, -1, dtype=np.int64)
    source_start = 0
    for output_index, (from_value, to_value) in enumerate(zip(output_from, output_to)):
        while source_start < source_from.size and source_to[source_start] <= from_value:
            source_start += 1
        candidate = source_start
        best_overlap = 0.0
        while candidate < source_from.size and source_from[candidate] < to_value:
            overlap = min(to_value, source_to[candidate]) - max(from_value, source_from[candidate])
            if overlap > best_overlap:
                best_overlap = overlap
                positions[output_index] = candidate
            candidate += 1
    return positions


def longest_overlap_positions_batch(groups: list[tuple[npt.NDArray[np.float64], ...]]) -> list[npt.NDArray[np.int64]]:
    """Select longest overlaps for independent groups in one native call."""
    if _rust_native is not None:
        native_groups = [
            tuple(np.ascontiguousarray(array, dtype=np.float64) for array in group)
            for group in groups
        ]
        return _rust_native.longest_overlap_positions_parallel(native_groups)
    announce_fallback()
    return [longest_overlap_positions(*group) for group in groups]


if njit is not None:

    @njit(cache=True, nogil=True)
    def _overlay_events_numba(
        event_measure_true: npt.NDArray[np.float64],
        event_measure_slk: npt.NDArray[np.float64],
        event_type: npt.NDArray[np.int64],
        event_df_num: npt.NDArray[np.int64],
        event_original_index: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64], int]:
        capacity = max(event_measure_true.size - 1, 0)
        result_true_from = np.empty(capacity, dtype=np.float64)
        result_true_to = np.empty(capacity, dtype=np.float64)
        result_slk_from = np.empty(capacity, dtype=np.float64)
        result_slk_to = np.empty(capacity, dtype=np.float64)
        result_original_index = np.full(capacity, -1, dtype=np.int64)
        result_additional_index = np.full(capacity, -1, dtype=np.int64)
        count = 0
        original_index = -1
        additional_index = -1
        last_true = 0.0
        last_slk = 0.0

        for row_index in range(event_measure_true.size):
            current_true = event_measure_true[row_index]
            current_slk = event_measure_slk[row_index]
            if row_index > 0 and current_true - last_true > 0 and current_slk - last_slk > 0 and (original_index >= 0 or additional_index >= 0):
                result_true_from[count] = last_true
                result_true_to[count] = current_true
                result_slk_from[count] = last_slk
                result_slk_to[count] = current_slk
                result_original_index[count] = original_index
                result_additional_index[count] = additional_index
                count += 1

            if event_type[row_index] == 0:
                if event_df_num[row_index] == 0:
                    original_index = event_original_index[row_index]
                else:
                    additional_index = event_original_index[row_index]
            else:
                if event_df_num[row_index] == 0:
                    original_index = -1
                else:
                    additional_index = -1

            last_true = current_true
            last_slk = current_slk

        return (
            result_true_from,
            result_true_to,
            result_slk_from,
            result_slk_to,
            result_original_index,
            result_additional_index,
            count,
        )

else:
    _overlay_events_numba = None


if njit is not None:

    @njit(cache=True, nogil=True)
    def _cross_sections_numba(
        event_measure_true: npt.NDArray[np.float64],
        event_measure_slk: npt.NDArray[np.float64],
        event_type: npt.NDArray[np.int64],
        event_path: npt.NDArray[np.int64],
        event_source: npt.NDArray[np.int64],
        output_true_from: npt.NDArray[np.float64],
        output_true_to: npt.NDArray[np.float64],
        output_slk_from: npt.NDArray[np.float64],
        output_slk_to: npt.NDArray[np.float64],
        output_path: npt.NDArray[np.int64],
        output_source: npt.NDArray[np.int64],
        output_overlap: npt.NDArray[np.float64],
        output_section: npt.NDArray[np.int64],
    ) -> int:
        event_count = event_measure_true.size
        path_count = 0
        for path in event_path:
            if path + 1 > path_count:
                path_count = path + 1

        active_source = np.zeros(event_count, dtype=np.uint8)
        source_path = np.full(event_count, -1, dtype=np.int64)
        source_previous = np.full(event_count, -1, dtype=np.int64)
        source_next = np.full(event_count, -1, dtype=np.int64)
        path_head = np.full(path_count, -1, dtype=np.int64)
        path_tail = np.full(path_count, -1, dtype=np.int64)
        active_paths = np.empty(path_count, dtype=np.int64)
        active_path_count = 0
        section_paths = np.empty(path_count, dtype=np.int64)
        section_path_count = 0
        overlap = np.zeros(event_count, dtype=np.float64)
        output_count = 0
        section_number = 0
        section_true_from = 0.0
        section_slk_from = 0.0
        section_true_to = 0.0
        section_slk_to = 0.0

        for event_index in range(event_count):
            if event_index > 0:
                interval_true_from = event_measure_true[event_index - 1]
                interval_true_to = event_measure_true[event_index]
                interval_slk_from = event_measure_slk[event_index - 1]
                interval_slk_to = event_measure_slk[event_index]
                if interval_true_to > interval_true_from and interval_slk_to > interval_slk_from and active_path_count > 0:
                    same_paths = active_path_count == section_path_count
                    if same_paths:
                        for path_position in range(active_path_count):
                            if active_paths[path_position] != section_paths[path_position]:
                                same_paths = False
                                break

                    if section_path_count == 0 or not same_paths:
                        if section_path_count > 0:
                            for path_position in range(section_path_count):
                                path = section_paths[path_position]
                                for source in range(event_count):
                                    if source_path[source] == path and overlap[source] > 0:
                                        if output_count < output_true_from.size:
                                            output_true_from[output_count] = section_true_from
                                            output_true_to[output_count] = section_true_to
                                            output_slk_from[output_count] = section_slk_from
                                            output_slk_to[output_count] = section_slk_to
                                            output_path[output_count] = path
                                            output_source[output_count] = source
                                            output_overlap[output_count] = overlap[source]
                                            output_section[output_count] = section_number
                                        output_count += 1
                            section_number += 1
                        section_true_from = interval_true_from
                        section_slk_from = interval_slk_from
                        section_path_count = active_path_count
                        for path_position in range(active_path_count):
                            section_paths[path_position] = active_paths[path_position]
                        overlap[:] = 0.0

                    section_true_to = interval_true_to
                    section_slk_to = interval_slk_to
                    for path_position in range(active_path_count):
                        path = active_paths[path_position]
                        source = path_head[path]
                        while source >= 0:
                            overlap[source] += interval_true_to - interval_true_from
                            source = source_next[source]

            path = event_path[event_index]
            source = event_source[event_index]
            source_path[source] = path
            if event_type[event_index] == 0:
                active_source[source] = 1
                source_previous[source] = path_tail[path]
                source_next[source] = -1
                if path_tail[path] >= 0:
                    source_next[path_tail[path]] = source
                else:
                    path_head[path] = source
                path_tail[path] = source
                if path_head[path] == source:
                    active_paths[active_path_count] = path
                    active_path_count += 1
            else:
                active_source[source] = 0
                previous_source = source_previous[source]
                next_source = source_next[source]
                if previous_source >= 0:
                    source_next[previous_source] = next_source
                else:
                    path_head[path] = next_source
                if next_source >= 0:
                    source_previous[next_source] = previous_source
                else:
                    path_tail[path] = previous_source
                if path_head[path] < 0:
                    for path_position in range(active_path_count):
                        if active_paths[path_position] == path:
                            for move_position in range(path_position, active_path_count - 1):
                                active_paths[move_position] = active_paths[move_position + 1]
                            active_path_count -= 1
                            break

        if section_path_count > 0:
            for path_position in range(section_path_count):
                path = section_paths[path_position]
                for source in range(event_count):
                    if source_path[source] == path and overlap[source] > 0:
                        if output_count < output_true_from.size:
                            output_true_from[output_count] = section_true_from
                            output_true_to[output_count] = section_true_to
                            output_slk_from[output_count] = section_slk_from
                            output_slk_to[output_count] = section_slk_to
                            output_path[output_count] = path
                            output_source[output_count] = source
                            output_overlap[output_count] = overlap[source]
                            output_section[output_count] = section_number
                        output_count += 1
        return output_count

else:
    _cross_sections_numba = None


def overlay_events(
    event_measure_true: npt.NDArray[np.float64],
    event_measure_slk: npt.NDArray[np.float64],
    event_type: npt.NDArray[np.int64],
    event_df_num: npt.NDArray[np.int64],
    event_original_index: npt.NDArray[np.int64],
) -> Any:
    """Run one sorted segmentation-overlay event sweep."""
    if _rust_native is not None:
        arrays = (
            np.ascontiguousarray(event_measure_true, dtype=np.float64),
            np.ascontiguousarray(event_measure_slk, dtype=np.float64),
            np.ascontiguousarray(event_type, dtype=np.int64),
            np.ascontiguousarray(event_df_num, dtype=np.int64),
            np.ascontiguousarray(event_original_index, dtype=np.int64),
        )
        return _rust_native.overlay_events(*arrays)
    if _overlay_events_numba is not None:
        result = _overlay_events_numba(event_measure_true, event_measure_slk, event_type, event_df_num, event_original_index)
        count = result[-1]
        return tuple(values[:count] for values in result[:-1])  # pyright: ignore[reportReturnType]

    output = [[] for _ in range(6)]
    original_index = -1
    announce_fallback()
    additional_index = -1
    last_true = None
    last_slk = None
    for current_true, current_slk, current_type, current_df_num, current_index in zip(
        event_measure_true, event_measure_slk, event_type, event_df_num, event_original_index
    ):
        if last_true is not None and current_true - last_true > 0 and current_slk - last_slk > 0 and (original_index >= 0 or additional_index >= 0):
            output[0].append(last_true)
            output[1].append(current_true)
            output[2].append(last_slk)
            output[3].append(current_slk)
            output[4].append(original_index)
            output[5].append(additional_index)
        if current_type == 0:
            if current_df_num == 0:
                original_index = current_index
            else:
                additional_index = current_index
        else:
            if current_df_num == 0:
                original_index = -1
            else:
                additional_index = -1
        last_true = current_true
        last_slk = current_slk
    return tuple(np.asarray(values) for values in output)
