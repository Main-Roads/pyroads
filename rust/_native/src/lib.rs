use std::cmp::Ordering;

use numpy::{
    PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rayon::prelude::*;

type OverlapResult = (Vec<i64>, Vec<i64>, Vec<f64>);

const AGG_AVERAGE: i64 = 3;
const AGG_LENGTH_WEIGHTED_AVERAGE: i64 = 4;
const AGG_LENGTH_WEIGHTED_PERCENTILE: i64 = 5;
const AGG_FIRST: i64 = 6;
const AGG_SUM_PROPORTION_OF_DATA: i64 = 7;
const AGG_SUM_PROPORTION_OF_TARGET: i64 = 8;
const AGG_SUM: i64 = 9;
const AGG_INDEX_OF_MAX: i64 = 10;
const AGG_INDEX_OF_MIN: i64 = 11;
const AGG_MIN: i64 = 12;
const AGG_MAX: i64 = 13;
const AGG_KEEP_LONGEST_SEGMENT: i64 = 1;
const AGG_KEEP_LONGEST: i64 = 2;

fn weighted_percentile(values: &[f64], weights: &[f64], percentile: f64) -> f64 {
    let mut pairs: Vec<(f64, f64)> = values
        .iter()
        .zip(weights.iter())
        .filter_map(|(&value, &weight)| {
            if value.is_nan() || weight <= 0.0 {
                None
            } else {
                Some((value, weight))
            }
        })
        .collect();
    if pairs.is_empty() {
        return f64::NAN;
    }
    if pairs.len() == 1 {
        return pairs[0].0;
    }
    pairs.sort_unstable_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let mut coordinates = vec![0.0; pairs.len()];
    let mut cumulative = 0.0;
    for index in 1..pairs.len() {
        cumulative += (pairs[index - 1].1 + pairs[index].1) * 0.5;
        coordinates[index] = cumulative;
    }
    let total = coordinates[pairs.len() - 1];
    if total <= 0.0 {
        return pairs[pairs.len() - 1].0;
    }
    for coordinate in &mut coordinates {
        *coordinate /= total;
    }
    if percentile <= coordinates[0] {
        return pairs[0].0;
    }
    if percentile >= coordinates[pairs.len() - 1] {
        return pairs[pairs.len() - 1].0;
    }
    for index in 1..pairs.len() {
        if coordinates[index] >= percentile {
            let x0 = coordinates[index - 1];
            let x1 = coordinates[index];
            if x1 == x0 {
                return pairs[index - 1].0;
            }
            let fraction = (percentile - x0) / (x1 - x0);
            return pairs[index - 1].0 + fraction * (pairs[index].0 - pairs[index - 1].0);
        }
    }
    pairs[pairs.len() - 1].0
}

fn aggregate_target(
    values: &[f64],
    overlaps: &[f64],
    data_lengths: &[f64],
    original_indices: &[i64],
    target_length: f64,
    agg_type: i64,
    percentile: f64,
) -> f64 {
    match agg_type {
        AGG_AVERAGE => {
            let valid: Vec<f64> = values
                .iter()
                .copied()
                .filter(|value| !value.is_nan())
                .collect();
            if valid.is_empty() {
                f64::NAN
            } else {
                valid.iter().sum::<f64>() / valid.len() as f64
            }
        }
        AGG_LENGTH_WEIGHTED_AVERAGE => {
            let (sum, weight) = values.iter().zip(overlaps).fold(
                (0.0, 0.0),
                |(sum, weight), (&value, &overlap)| {
                    if value.is_nan() || overlap <= 0.0 {
                        (sum, weight)
                    } else {
                        (sum + value * overlap, weight + overlap)
                    }
                },
            );
            if weight == 0.0 {
                f64::NAN
            } else {
                sum / weight
            }
        }
        AGG_LENGTH_WEIGHTED_PERCENTILE => weighted_percentile(values, overlaps, percentile),
        AGG_FIRST => values
            .iter()
            .zip(original_indices)
            .filter(|(value, _)| !value.is_nan())
            .min_by_key(|(_, index)| *index)
            .map(|(value, _)| *value)
            .unwrap_or(f64::NAN),
        AGG_SUM => {
            let mut sum = 0.0;
            let mut found = false;
            for &value in values {
                if !value.is_nan() {
                    sum += value;
                    found = true;
                }
            }
            if found { sum } else { f64::NAN }
        }
        AGG_SUM_PROPORTION_OF_DATA => {
            let mut sum = 0.0;
            let mut found = false;
            for ((&value, &overlap), &length) in values.iter().zip(overlaps).zip(data_lengths) {
                if !value.is_nan() && length > 0.0 {
                    sum += value * overlap / length;
                    found = true;
                }
            }
            if found { sum } else { f64::NAN }
        }
        AGG_SUM_PROPORTION_OF_TARGET => {
            if target_length <= 0.0 {
                return f64::NAN;
            }
            let mut sum = 0.0;
            let mut found = false;
            for (&value, &overlap) in values.iter().zip(overlaps) {
                if !value.is_nan() {
                    sum += value * overlap;
                    found = true;
                }
            }
            if found { sum / target_length } else { f64::NAN }
        }
        AGG_MAX | AGG_INDEX_OF_MAX => {
            let mut best_value = f64::NEG_INFINITY;
            let mut best_index = -1;
            for (&value, &index) in values.iter().zip(original_indices) {
                if !value.is_nan() && value > best_value {
                    best_value = value;
                    best_index = index;
                }
            }
            if best_index < 0 {
                f64::NAN
            } else if agg_type == AGG_MAX {
                best_value
            } else {
                best_index as f64
            }
        }
        AGG_MIN | AGG_INDEX_OF_MIN => {
            let mut best_value = f64::INFINITY;
            let mut best_index = -1;
            for (&value, &index) in values.iter().zip(original_indices) {
                if !value.is_nan() && value < best_value {
                    best_value = value;
                    best_index = index;
                }
            }
            if best_index < 0 {
                f64::NAN
            } else if agg_type == AGG_MIN {
                best_value
            } else {
                best_index as f64
            }
        }
        AGG_KEEP_LONGEST_SEGMENT | AGG_KEEP_LONGEST => {
            let mut best_overlap = -1.0;
            let mut best_value = f64::NAN;
            for (&value, &overlap) in values.iter().zip(overlaps) {
                if !value.is_nan() && overlap > best_overlap {
                    best_overlap = overlap;
                    best_value = value;
                }
            }
            best_value
        }
        _ => f64::NAN,
    }
}

#[pyfunction]
fn cumulative_p<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_slice()?;
    if data.len() < 2 {
        return Ok(PyArray1::from_vec(py, Vec::new()).unbind());
    }
    let n = data.len() - 1;
    let mut left_sum = vec![0.0; n];
    let mut right_sum = vec![0.0; n];
    let mut left_square_sum = vec![0.0; n];
    let mut right_square_sum = vec![0.0; n];
    let mut sum = 0.0;
    let mut square_sum = 0.0;
    for index in 0..n {
        sum += data[index];
        square_sum += data[index] * data[index];
        left_sum[index] = sum;
        left_square_sum[index] = square_sum;
    }
    sum = 0.0;
    square_sum = 0.0;
    for index in (0..n).rev() {
        let data_index = index + 1;
        sum += data[data_index];
        square_sum += data[data_index] * data[data_index];
        right_sum[index] = sum;
        right_square_sum[index] = square_sum;
    }
    let result: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|index| {
            let left_n = (index + 1) as f64;
            let right_n = (n - index) as f64;
            let left = ((left_n * left_square_sum[index] / (left_sum[index] * left_sum[index])
                - 1.0)
                * left_n
                / (left_n - 1.0))
                .sqrt();
            let right =
                ((right_n * right_square_sum[index] / (right_sum[index] * right_sum[index]) - 1.0)
                    * right_n
                    / (right_n - 1.0))
                    .sqrt();
            (left + right) / 2.0
        })
        .collect();
    Ok(PyArray1::from_vec(py, result).unbind())
}

#[pyfunction]
fn cumulative_q<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let data = data.as_slice()?;
    if data.len() < 2 {
        return Ok(PyArray1::from_vec(py, Vec::new()).unbind());
    }
    let n = data.len() - 1;
    let mut left_sum = vec![0.0; n];
    let mut right_sum = vec![0.0; n];
    let mut left_square_sum = vec![0.0; n];
    let mut right_square_sum = vec![0.0; n];
    let mut sum = 0.0;
    let mut square_sum = 0.0;
    for index in 0..n {
        sum += data[index];
        square_sum += data[index] * data[index];
        left_sum[index] = sum;
        left_square_sum[index] = square_sum;
    }
    sum = 0.0;
    square_sum = 0.0;
    for index in (0..n).rev() {
        let data_index = index + 1;
        sum += data[data_index];
        square_sum += data[data_index] * data[data_index];
        right_sum[index] = sum;
        right_square_sum[index] = square_sum;
    }
    let total_sum: f64 = data.iter().sum();
    let total_square_sum: f64 = data.iter().map(|value| value * value).sum();
    let denominator = total_square_sum - total_sum * total_sum / data.len() as f64;
    let result: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|index| {
            let left_n = (index + 1) as f64;
            let right_n = (n - index) as f64;
            1.0 - ((left_square_sum[index] - left_sum[index] * left_sum[index] / left_n)
                + (right_square_sum[index] - right_sum[index] * right_sum[index] / right_n))
                / denominator
        })
        .collect();
    Ok(PyArray1::from_vec(py, result).unbind())
}

fn longest_overlap_positions_impl(
    output_from: &[f64],
    output_to: &[f64],
    source_from: &[f64],
    source_to: &[f64],
) -> Vec<i64> {
    let mut positions = vec![-1_i64; output_from.len()];
    let mut source_start = 0;
    for output_index in 0..output_from.len() {
        while source_start < source_from.len()
            && source_to[source_start] <= output_from[output_index]
        {
            source_start += 1;
        }
        let mut candidate = source_start;
        let mut best_overlap = 0.0;
        while candidate < source_from.len() && source_from[candidate] < output_to[output_index] {
            let overlap = output_to[output_index].min(source_to[candidate])
                - output_from[output_index].max(source_from[candidate]);
            if overlap > best_overlap {
                best_overlap = overlap;
                positions[output_index] = candidate as i64;
            }
            candidate += 1;
        }
    }
    positions
}

#[pyfunction]
fn longest_overlap_positions<'py>(
    py: Python<'py>,
    output_from: PyReadonlyArray1<'py, f64>,
    output_to: PyReadonlyArray1<'py, f64>,
    source_from: PyReadonlyArray1<'py, f64>,
    source_to: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<i64>>> {
    let output_from = output_from.as_slice()?;
    let output_to = output_to.as_slice()?;
    let source_from = source_from.as_slice()?;
    let source_to = source_to.as_slice()?;
    if output_from.len() != output_to.len() || source_from.len() != source_to.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "interval start and end arrays must have matching lengths",
        ));
    }
    let positions = longest_overlap_positions_impl(output_from, output_to, source_from, source_to);
    Ok(PyArray1::from_vec(py, positions).unbind())
}

#[pyfunction]
fn longest_overlap_positions_parallel<'py>(
    py: Python<'py>,
    groups: &Bound<'py, PyList>,
) -> PyResult<Vec<Py<PyArray1<i64>>>> {
    let owned_groups: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = groups
        .iter()
        .map(|group| {
            let tuple = group.cast::<PyTuple>()?;
            if tuple.len() != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "each overlap group must contain four arrays",
                ));
            }
            let output_from = tuple.get_item(0)?.extract::<PyReadonlyArray1<'_, f64>>()?;
            let output_to = tuple.get_item(1)?.extract::<PyReadonlyArray1<'_, f64>>()?;
            let source_from = tuple.get_item(2)?.extract::<PyReadonlyArray1<'_, f64>>()?;
            let source_to = tuple.get_item(3)?.extract::<PyReadonlyArray1<'_, f64>>()?;
            let output_from = output_from.as_slice()?.to_vec();
            let output_to = output_to.as_slice()?.to_vec();
            let source_from = source_from.as_slice()?.to_vec();
            let source_to = source_to.as_slice()?.to_vec();
            if output_from.len() != output_to.len() || source_from.len() != source_to.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "interval start and end arrays must have matching lengths",
                ));
            }
            Ok((output_from, output_to, source_from, source_to))
        })
        .collect::<PyResult<_>>()?;

    let results = py.detach(|| {
        owned_groups
            .par_iter()
            .map(|(output_from, output_to, source_from, source_to)| {
                longest_overlap_positions_impl(output_from, output_to, source_from, source_to)
            })
            .collect::<Vec<_>>()
    });
    Ok(results
        .into_iter()
        .map(|positions| PyArray1::from_vec(py, positions).unbind())
        .collect())
}

#[pyfunction]
fn overlay_events<'py>(
    py: Python<'py>,
    event_measure_true: PyReadonlyArray1<'py, f64>,
    event_measure_slk: PyReadonlyArray1<'py, f64>,
    event_type: PyReadonlyArray1<'py, i64>,
    event_df_num: PyReadonlyArray1<'py, i64>,
    event_original_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
)> {
    let event_measure_true = event_measure_true.as_slice()?;
    let event_measure_slk = event_measure_slk.as_slice()?;
    let event_type = event_type.as_slice()?;
    let event_df_num = event_df_num.as_slice()?;
    let event_original_index = event_original_index.as_slice()?;
    let event_count = event_measure_true.len();
    if event_measure_slk.len() != event_count
        || event_type.len() != event_count
        || event_df_num.len() != event_count
        || event_original_index.len() != event_count
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "overlay event arrays must have matching lengths",
        ));
    }

    let capacity = event_count.saturating_sub(1);
    let mut result_true_from = Vec::with_capacity(capacity);
    let mut result_true_to = Vec::with_capacity(capacity);
    let mut result_slk_from = Vec::with_capacity(capacity);
    let mut result_slk_to = Vec::with_capacity(capacity);
    let mut result_original_index = Vec::with_capacity(capacity);
    let mut result_additional_index = Vec::with_capacity(capacity);
    let mut original_index = -1_i64;
    let mut additional_index = -1_i64;
    let mut last_true = 0.0;
    let mut last_slk = 0.0;

    for row_index in 0..event_count {
        let current_true = event_measure_true[row_index];
        let current_slk = event_measure_slk[row_index];
        if row_index > 0
            && current_true - last_true > 0.0
            && current_slk - last_slk > 0.0
            && (original_index >= 0 || additional_index >= 0)
        {
            result_true_from.push(last_true);
            result_true_to.push(current_true);
            result_slk_from.push(last_slk);
            result_slk_to.push(current_slk);
            result_original_index.push(original_index);
            result_additional_index.push(additional_index);
        }

        if event_type[row_index] == 0 {
            if event_df_num[row_index] == 0 {
                original_index = event_original_index[row_index];
            } else {
                additional_index = event_original_index[row_index];
            }
        } else if event_df_num[row_index] == 0 {
            original_index = -1;
        } else {
            additional_index = -1;
        }

        last_true = current_true;
        last_slk = current_slk;
    }

    Ok((
        PyArray1::from_vec(py, result_true_from).unbind(),
        PyArray1::from_vec(py, result_true_to).unbind(),
        PyArray1::from_vec(py, result_slk_from).unbind(),
        PyArray1::from_vec(py, result_slk_to).unbind(),
        PyArray1::from_vec(py, result_original_index).unbind(),
        PyArray1::from_vec(py, result_additional_index).unbind(),
    ))
}

#[pyfunction]
fn cross_sections(
    event_measure_true: PyReadonlyArray1<'_, f64>,
    event_measure_slk: PyReadonlyArray1<'_, f64>,
    event_type: PyReadonlyArray1<'_, i64>,
    event_path: PyReadonlyArray1<'_, i64>,
    event_source: PyReadonlyArray1<'_, i64>,
    mut output_true_from: PyReadwriteArray1<'_, f64>,
    mut output_true_to: PyReadwriteArray1<'_, f64>,
    mut output_slk_from: PyReadwriteArray1<'_, f64>,
    mut output_slk_to: PyReadwriteArray1<'_, f64>,
    mut output_path: PyReadwriteArray1<'_, i64>,
    mut output_source: PyReadwriteArray1<'_, i64>,
    mut output_overlap: PyReadwriteArray1<'_, f64>,
    mut output_section: PyReadwriteArray1<'_, i64>,
) -> PyResult<usize> {
    let event_measure_true = event_measure_true.as_slice()?;
    let event_measure_slk = event_measure_slk.as_slice()?;
    let event_type = event_type.as_slice()?;
    let event_path = event_path.as_slice()?;
    let event_source = event_source.as_slice()?;
    let event_count = event_measure_true.len();
    if event_measure_slk.len() != event_count
        || event_type.len() != event_count
        || event_path.len() != event_count
        || event_source.len() != event_count
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cross-section event arrays must have matching lengths",
        ));
    }

    let output_true_from = output_true_from.as_slice_mut()?;
    let output_true_to = output_true_to.as_slice_mut()?;
    let output_slk_from = output_slk_from.as_slice_mut()?;
    let output_slk_to = output_slk_to.as_slice_mut()?;
    let output_path = output_path.as_slice_mut()?;
    let output_source = output_source.as_slice_mut()?;
    let output_overlap = output_overlap.as_slice_mut()?;
    let output_section = output_section.as_slice_mut()?;
    let output_capacity = output_true_from.len();
    if output_true_to.len() != output_capacity
        || output_slk_from.len() != output_capacity
        || output_slk_to.len() != output_capacity
        || output_path.len() != output_capacity
        || output_source.len() != output_capacity
        || output_overlap.len() != output_capacity
        || output_section.len() != output_capacity
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cross-section output arrays must have matching lengths",
        ));
    }

    let path_count = event_path
        .iter()
        .copied()
        .filter(|&path| path >= 0)
        .max()
        .map_or(0, |path| path as usize + 1);
    let mut active_source = vec![false; event_count];
    let mut source_path = vec![-1_i64; event_count];
    let mut source_previous = vec![-1_i64; event_count];
    let mut source_next = vec![-1_i64; event_count];
    let mut path_head = vec![-1_i64; path_count];
    let mut path_tail = vec![-1_i64; path_count];
    let mut active_paths = vec![0_i64; path_count];
    let mut active_path_count = 0;
    let mut section_paths = vec![0_i64; path_count];
    let mut section_path_count = 0;
    let mut overlap = vec![0.0; event_count];
    let mut output_count = 0;
    let mut section_number = 0_i64;
    let mut section_true_from = 0.0;
    let mut section_slk_from = 0.0;
    let mut section_true_to = 0.0;
    let mut section_slk_to = 0.0;

    for event_index in 0..event_count {
        if event_index > 0 {
            let interval_true_from = event_measure_true[event_index - 1];
            let interval_true_to = event_measure_true[event_index];
            let interval_slk_from = event_measure_slk[event_index - 1];
            let interval_slk_to = event_measure_slk[event_index];
            if interval_true_to > interval_true_from
                && interval_slk_to > interval_slk_from
                && active_path_count > 0
            {
                let same_paths = section_path_count == active_path_count
                    && (0..active_path_count)
                        .all(|position| active_paths[position] == section_paths[position]);
                if section_path_count == 0 || !same_paths {
                    if section_path_count > 0 {
                        for path_position in 0..section_path_count {
                            let path = section_paths[path_position];
                            for source in 0..event_count {
                                if source_path[source] == path && overlap[source] > 0.0 {
                                    if output_count < output_capacity {
                                        output_true_from[output_count] = section_true_from;
                                        output_true_to[output_count] = section_true_to;
                                        output_slk_from[output_count] = section_slk_from;
                                        output_slk_to[output_count] = section_slk_to;
                                        output_path[output_count] = path;
                                        output_source[output_count] = source as i64;
                                        output_overlap[output_count] = overlap[source];
                                        output_section[output_count] = section_number;
                                    }
                                    output_count += 1;
                                }
                            }
                        }
                        section_number += 1;
                    }
                    section_true_from = interval_true_from;
                    section_slk_from = interval_slk_from;
                    section_path_count = active_path_count;
                    section_paths[..active_path_count]
                        .copy_from_slice(&active_paths[..active_path_count]);
                    overlap.fill(0.0);
                }

                section_true_to = interval_true_to;
                section_slk_to = interval_slk_to;
                for path_position in 0..active_path_count {
                    let path = active_paths[path_position] as usize;
                    let mut source = path_head[path];
                    while source >= 0 {
                        overlap[source as usize] += interval_true_to - interval_true_from;
                        source = source_next[source as usize];
                    }
                }
            }
        }

        let path = event_path[event_index] as usize;
        let source = event_source[event_index] as usize;
        source_path[source] = path as i64;
        if event_type[event_index] == 0 {
            active_source[source] = true;
            source_previous[source] = path_tail[path];
            source_next[source] = -1;
            if path_tail[path] >= 0 {
                source_next[path_tail[path] as usize] = source as i64;
            } else {
                path_head[path] = source as i64;
            }
            path_tail[path] = source as i64;
            if path_head[path] == source as i64 {
                active_paths[active_path_count] = path as i64;
                active_path_count += 1;
            }
        } else {
            active_source[source] = false;
            let previous_source = source_previous[source];
            let next_source = source_next[source];
            if previous_source >= 0 {
                source_next[previous_source as usize] = next_source;
            } else {
                path_head[path] = next_source;
            }
            if next_source >= 0 {
                source_previous[next_source as usize] = previous_source;
            } else {
                path_tail[path] = previous_source;
            }
            if path_head[path] < 0 {
                if let Some(path_position) = active_paths[..active_path_count]
                    .iter()
                    .position(|&active_path| active_path == path as i64)
                {
                    active_paths.copy_within(path_position + 1..active_path_count, path_position);
                    active_path_count -= 1;
                }
            }
        }
    }

    if section_path_count > 0 {
        for path_position in 0..section_path_count {
            let path = section_paths[path_position];
            for source in 0..event_count {
                if source_path[source] == path && overlap[source] > 0.0 {
                    if output_count < output_capacity {
                        output_true_from[output_count] = section_true_from;
                        output_true_to[output_count] = section_true_to;
                        output_slk_from[output_count] = section_slk_from;
                        output_slk_to[output_count] = section_slk_to;
                        output_path[output_count] = path;
                        output_source[output_count] = source as i64;
                        output_overlap[output_count] = overlap[source];
                        output_section[output_count] = section_number;
                    }
                    output_count += 1;
                }
            }
        }
    }
    Ok(output_count)
}

fn find_overlaps(
    tgt_starts: &[f64],
    tgt_ends: &[f64],
    data_starts: &[f64],
    data_ends: &[f64],
) -> OverlapResult {
    let mut data_order: Vec<usize> = (0..data_ends.len()).collect();
    data_order.sort_unstable_by(|&left, &right| {
        data_ends[left]
            .partial_cmp(&data_ends[right])
            .unwrap_or(Ordering::Equal)
    });
    let capacity = tgt_starts.len().saturating_mul(5).max(1000);
    let mut target_indices = Vec::with_capacity(capacity);
    let mut data_indices = Vec::with_capacity(capacity);
    let mut overlap_lengths = Vec::with_capacity(capacity);

    for (target_index, (&target_start, &target_end)) in
        tgt_starts.iter().zip(tgt_ends.iter()).enumerate()
    {
        if target_end <= target_start {
            continue;
        }
        let mut low = 0;
        let mut high = data_order.len();
        while low < high {
            let middle = (low + high) / 2;
            if data_ends[data_order[middle]] <= target_start {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        for &data_index in &data_order[low..] {
            let data_start = data_starts[data_index];
            let data_end = data_ends[data_index];
            if data_start >= target_end {
                continue;
            }
            let overlap_length = target_end.min(data_end) - target_start.max(data_start);
            if overlap_length > 0.0 {
                target_indices.push(target_index as i64);
                data_indices.push(data_index as i64);
                overlap_lengths.push(overlap_length);
            }
        }
    }
    (target_indices, data_indices, overlap_lengths)
}

#[pyfunction]
fn find_overlapping_intervals<'py>(
    py: Python<'py>,
    tgt_starts: PyReadonlyArray1<'py, f64>,
    tgt_ends: PyReadonlyArray1<'py, f64>,
    data_starts: PyReadonlyArray1<'py, f64>,
    data_ends: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Py<PyArray1<i64>>, Py<PyArray1<i64>>, Py<PyArray1<f64>>)> {
    let tgt_starts = tgt_starts.as_slice()?;
    let tgt_ends = tgt_ends.as_slice()?;
    let data_starts = data_starts.as_slice()?;
    let data_ends = data_ends.as_slice()?;

    if tgt_starts.len() != tgt_ends.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target start and end arrays must have the same length",
        ));
    }
    if data_starts.len() != data_ends.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "data start and end arrays must have the same length",
        ));
    }

    let (target_indices, data_indices, overlap_lengths) =
        find_overlaps(tgt_starts, tgt_ends, data_starts, data_ends);

    Ok((
        PyArray1::from_vec(py, target_indices).unbind(),
        PyArray1::from_vec(py, data_indices).unbind(),
        PyArray1::from_vec(py, overlap_lengths).unbind(),
    ))
}

#[pyfunction]
fn find_overlapping_intervals_parallel<'py>(
    py: Python<'py>,
    groups: &Bound<'py, PyList>,
) -> PyResult<Vec<(Py<PyArray1<i64>>, Py<PyArray1<i64>>, Py<PyArray1<f64>>)>> {
    let owned_groups: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = groups
        .iter()
        .map(|group| {
            let tuple = group.cast::<PyTuple>()?;
            if tuple.len() != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "each overlap group must contain four arrays",
                ));
            }
            let target_starts = tuple.get_item(0)?.extract::<PyReadonlyArray1<'_, f64>>()?;
            let target_ends = tuple.get_item(1)?.extract::<PyReadonlyArray1<'_, f64>>()?;
            let data_starts = tuple.get_item(2)?.extract::<PyReadonlyArray1<'_, f64>>()?;
            let data_ends = tuple.get_item(3)?.extract::<PyReadonlyArray1<'_, f64>>()?;
            let target_starts = target_starts.as_slice()?.to_vec();
            let target_ends = target_ends.as_slice()?.to_vec();
            let data_starts = data_starts.as_slice()?.to_vec();
            let data_ends = data_ends.as_slice()?.to_vec();
            if target_starts.len() != target_ends.len() || data_starts.len() != data_ends.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "interval start and end arrays must have matching lengths",
                ));
            }
            Ok((target_starts, target_ends, data_starts, data_ends))
        })
        .collect::<PyResult<_>>()?;

    let results = py.detach(|| {
        owned_groups
            .par_iter()
            .map(|(target_starts, target_ends, data_starts, data_ends)| {
                find_overlaps(target_starts, target_ends, data_starts, data_ends)
            })
            .collect::<Vec<_>>()
    });

    Ok(results
        .into_iter()
        .map(|(target_indices, data_indices, overlap_lengths)| {
            (
                PyArray1::from_vec(py, target_indices).unbind(),
                PyArray1::from_vec(py, data_indices).unbind(),
                PyArray1::from_vec(py, overlap_lengths).unbind(),
            )
        })
        .collect())
}

#[pyfunction]
fn aggregate_all_targets_numeric<'py>(
    py: Python<'py>,
    n_targets: usize,
    target_indices: PyReadonlyArray1<'py, i64>,
    data_indices: PyReadonlyArray1<'py, i64>,
    overlap_lengths: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray1<'py, f64>,
    data_lengths: PyReadonlyArray1<'py, f64>,
    original_indices: PyReadonlyArray1<'py, i64>,
    target_lengths: PyReadonlyArray1<'py, f64>,
    agg_type: i64,
    percentile: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let target_indices = target_indices.as_slice()?;
    let data_indices = data_indices.as_slice()?;
    let overlap_lengths = overlap_lengths.as_slice()?;
    let values = values.as_slice()?;
    let data_lengths = data_lengths.as_slice()?;
    let original_indices = original_indices.as_slice()?;
    let target_lengths = target_lengths.as_slice()?;
    if target_indices.len() != data_indices.len() || target_indices.len() != overlap_lengths.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "overlap arrays must have matching lengths",
        ));
    }
    if values.len() != data_lengths.len() || values.len() != original_indices.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "data arrays must have matching lengths",
        ));
    }
    if target_lengths.len() != n_targets {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_lengths must match n_targets",
        ));
    }

    let mut offsets = vec![0usize; n_targets + 1];
    for &target_index in target_indices {
        if target_index < 0 || target_index as usize >= n_targets {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target index is out of bounds",
            ));
        }
        offsets[target_index as usize + 1] += 1;
    }
    for target in 0..n_targets {
        offsets[target + 1] += offsets[target];
    }
    let mut grouped_data_indices = vec![0usize; data_indices.len()];
    let mut grouped_overlaps = vec![0.0; overlap_lengths.len()];
    let mut positions = offsets[..n_targets].to_vec();
    for (position, (&target_index, &data_index)) in
        target_indices.iter().zip(data_indices).enumerate()
    {
        if data_index < 0 || data_index as usize >= values.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "data index is out of bounds",
            ));
        }
        let target = target_index as usize;
        let output_position = positions[target];
        grouped_data_indices[output_position] = data_index as usize;
        grouped_overlaps[output_position] = overlap_lengths[position];
        positions[target] += 1;
    }

    let mut results = vec![f64::NAN; n_targets];
    results
        .par_iter_mut()
        .enumerate()
        .for_each(|(target, result)| {
            let start = offsets[target];
            let end = offsets[target + 1];
            if start == end {
                return;
            }
            let mut target_values = Vec::with_capacity(end - start);
            let mut target_overlaps = Vec::with_capacity(end - start);
            let mut target_data_lengths = Vec::with_capacity(end - start);
            let mut target_original_indices = Vec::with_capacity(end - start);
            for position in start..end {
                let data_index = grouped_data_indices[position];
                target_values.push(values[data_index]);
                target_overlaps.push(grouped_overlaps[position]);
                target_data_lengths.push(data_lengths[data_index]);
                target_original_indices.push(original_indices[data_index]);
            }
            *result = aggregate_target(
                &target_values,
                &target_overlaps,
                &target_data_lengths,
                &target_original_indices,
                target_lengths[target],
                agg_type,
                percentile,
            );
        });

    Ok(PyArray1::from_vec(py, results).unbind())
}

#[pyfunction]
fn aggregate_all_targets_numeric_batch<'py>(
    py: Python<'py>,
    n_targets: usize,
    target_indices: PyReadonlyArray1<'py, i64>,
    data_indices: PyReadonlyArray1<'py, i64>,
    overlap_lengths: PyReadonlyArray1<'py, f64>,
    values: PyReadonlyArray2<'py, f64>,
    data_lengths: PyReadonlyArray1<'py, f64>,
    original_indices: PyReadonlyArray1<'py, i64>,
    target_lengths: PyReadonlyArray1<'py, f64>,
    agg_types: PyReadonlyArray1<'py, i64>,
    percentiles: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let target_indices = target_indices.as_slice()?;
    let data_indices = data_indices.as_slice()?;
    let overlap_lengths = overlap_lengths.as_slice()?;
    let value_shape = values.shape().to_vec();
    let values = values.as_slice()?;
    let data_lengths = data_lengths.as_slice()?;
    let original_indices = original_indices.as_slice()?;
    let target_lengths = target_lengths.as_slice()?;
    let agg_types = agg_types.as_slice()?;
    let percentiles = percentiles.as_slice()?;
    if value_shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "values must be a two-dimensional array",
        ));
    }
    let action_count = value_shape[0];
    let data_count = value_shape[1];
    if target_indices.len() != data_indices.len() || target_indices.len() != overlap_lengths.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "overlap arrays must have matching lengths",
        ));
    }
    if data_lengths.len() != data_count || original_indices.len() != data_count {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "data arrays must match the values column length",
        ));
    }
    if target_lengths.len() != n_targets
        || agg_types.len() != action_count
        || percentiles.len() != action_count
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "batch aggregation arrays have inconsistent lengths",
        ));
    }

    let mut offsets = vec![0usize; n_targets + 1];
    for &target_index in target_indices {
        if target_index < 0 || target_index as usize >= n_targets {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target index is out of bounds",
            ));
        }
        offsets[target_index as usize + 1] += 1;
    }
    for target in 0..n_targets {
        offsets[target + 1] += offsets[target];
    }
    let mut grouped_data_indices = vec![0usize; data_indices.len()];
    let mut grouped_overlaps = vec![0.0; overlap_lengths.len()];
    let mut positions = offsets[..n_targets].to_vec();
    for (position, (&target_index, &data_index)) in
        target_indices.iter().zip(data_indices).enumerate()
    {
        if data_index < 0 || data_index as usize >= data_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "data index is out of bounds",
            ));
        }
        let target = target_index as usize;
        let output_position = positions[target];
        grouped_data_indices[output_position] = data_index as usize;
        grouped_overlaps[output_position] = overlap_lengths[position];
        positions[target] += 1;
    }

    let mut results = vec![f64::NAN; action_count * n_targets];
    if n_targets > 0 {
        results
            .par_chunks_mut(n_targets)
            .enumerate()
            .for_each(|(action, action_results)| {
                let action_values = &values[action * data_count..(action + 1) * data_count];
                for target in 0..n_targets {
                    let start = offsets[target];
                    let end = offsets[target + 1];
                    if start == end {
                        continue;
                    }
                    let mut target_values = Vec::with_capacity(end - start);
                    let mut target_overlaps = Vec::with_capacity(end - start);
                    let mut target_data_lengths = Vec::with_capacity(end - start);
                    let mut target_original_indices = Vec::with_capacity(end - start);
                    for position in start..end {
                        let data_index = grouped_data_indices[position];
                        target_values.push(action_values[data_index]);
                        target_overlaps.push(grouped_overlaps[position]);
                        target_data_lengths.push(data_lengths[data_index]);
                        target_original_indices.push(original_indices[data_index]);
                    }
                    action_results[target] = aggregate_target(
                        &target_values,
                        &target_overlaps,
                        &target_data_lengths,
                        &target_original_indices,
                        target_lengths[target],
                        agg_types[action],
                        percentiles[action],
                    );
                }
            });
    }

    Ok(PyArray1::from_vec(py, results).unbind())
}

#[pyfunction]
fn aggregate_keep_longest_categorical<'py>(
    py: Python<'py>,
    n_targets: usize,
    target_indices: PyReadonlyArray1<'py, i64>,
    data_indices: PyReadonlyArray1<'py, i64>,
    overlap_lengths: PyReadonlyArray1<'py, f64>,
    codes: PyReadonlyArray1<'py, i64>,
) -> PyResult<Py<PyArray1<i64>>> {
    let target_indices = target_indices.as_slice()?;
    let data_indices = data_indices.as_slice()?;
    let overlap_lengths = overlap_lengths.as_slice()?;
    let codes = codes.as_slice()?;
    if target_indices.len() != data_indices.len() || target_indices.len() != overlap_lengths.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "overlap arrays must have matching lengths",
        ));
    }

    let mut offsets = vec![0usize; n_targets + 1];
    for &target_index in target_indices {
        if target_index < 0 || target_index as usize >= n_targets {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target index is out of bounds",
            ));
        }
        offsets[target_index as usize + 1] += 1;
    }
    for target in 0..n_targets {
        offsets[target + 1] += offsets[target];
    }
    let mut grouped_data_indices = vec![0usize; data_indices.len()];
    let mut grouped_overlaps = vec![0.0; overlap_lengths.len()];
    let mut positions = offsets[..n_targets].to_vec();
    for (position, (&target_index, &data_index)) in
        target_indices.iter().zip(data_indices).enumerate()
    {
        if data_index < 0 || data_index as usize >= codes.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "data index is out of bounds",
            ));
        }
        let target = target_index as usize;
        let output_position = positions[target];
        grouped_data_indices[output_position] = data_index as usize;
        grouped_overlaps[output_position] = overlap_lengths[position];
        positions[target] += 1;
    }

    let mut results = vec![-1_i64; n_targets];
    if n_targets > 0 {
        results
            .par_iter_mut()
            .enumerate()
            .for_each(|(target, result)| {
                let start = offsets[target];
                let end = offsets[target + 1];
                let mut totals: Vec<(i64, f64, usize)> = Vec::new();
                for position in start..end {
                    let code = codes[grouped_data_indices[position]];
                    if code < 0 {
                        continue;
                    }
                    if let Some((_, total, _)) =
                        totals.iter_mut().find(|(value, _, _)| *value == code)
                    {
                        *total += grouped_overlaps[position];
                    } else {
                        totals.push((code, grouped_overlaps[position], position - start));
                    }
                }
                let mut best_code = -1_i64;
                let mut best_total = -1.0;
                let mut best_order = usize::MAX;
                for (code, total, order) in totals {
                    let tied = (total - best_total).abs() <= 1e-8 + 1e-5 * best_total.abs();
                    if total > best_total || (tied && order < best_order) {
                        best_code = code;
                        best_total = total;
                        best_order = order;
                    }
                }
                *result = best_code;
            });
    }
    Ok(PyArray1::from_vec(py, results).unbind())
}

fn linspace_steps_impl(
    measure_from: f64,
    measure_to: f64,
    multiples: f64,
    minimum_length: f64,
) -> Vec<f64> {
    if measure_from > measure_to {
        return Vec::new();
    }
    let left = (measure_from / multiples).ceil();
    let right = (measure_to / multiples).floor();
    if right < left {
        return vec![measure_from, measure_to];
    }
    let count = (right - left) as usize;
    let mut result: Vec<f64> = (0..=count)
        .map(|index| (index as f64 + left) * multiples)
        .collect();
    if result.is_empty() {
        return vec![measure_from, measure_to];
    }
    if result[0] != measure_from {
        if ((result[0] - measure_from) * 1e6).round() / 1e6 < minimum_length {
            result[0] = measure_from;
        } else {
            result.insert(0, measure_from);
        }
    }
    let last = result.len() - 1;
    if result[last] != measure_to {
        if ((measure_to - result[last]) * 1e6).round() / 1e6 < minimum_length {
            result[last] = measure_to;
        } else {
            result.push(measure_to);
        }
    }
    result
}

#[pyfunction]
fn linspace_steps_batch<'py>(
    py: Python<'py>,
    measure_from: PyReadonlyArray1<'py, f64>,
    measure_to: PyReadonlyArray1<'py, f64>,
    multiples: PyReadonlyArray1<'py, f64>,
    minimum_lengths: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<i64>>)> {
    let measure_from = measure_from.as_slice()?;
    let measure_to = measure_to.as_slice()?;
    let multiples = multiples.as_slice()?;
    let minimum_lengths = minimum_lengths.as_slice()?;
    let count = measure_from.len();
    if measure_to.len() != count || multiples.len() != count || minimum_lengths.len() != count {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "linspace batch arrays must have matching lengths",
        ));
    }
    let rows: Vec<Vec<f64>> = (0..count)
        .into_par_iter()
        .map(|index| {
            linspace_steps_impl(
                measure_from[index],
                measure_to[index],
                multiples[index],
                minimum_lengths[index],
            )
        })
        .collect();
    let mut values = Vec::new();
    let mut offsets = Vec::with_capacity(count + 1);
    offsets.push(0_i64);
    for row in rows {
        values.extend(row);
        offsets.push(values.len() as i64);
    }
    Ok((
        PyArray1::from_vec(py, values).unbind(),
        PyArray1::from_vec(py, offsets).unbind(),
    ))
}

#[pyfunction]
fn fixed_segment_boundaries_batch<'py>(
    py: Python<'py>,
    measure_from: PyReadonlyArray1<'py, f64>,
    measure_to: PyReadonlyArray1<'py, f64>,
    segment_length: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<i64>>)> {
    let measure_from = measure_from.as_slice()?;
    let measure_to = measure_to.as_slice()?;
    if measure_from.len() != measure_to.len() || segment_length <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fixed segment arrays must match and segment_length must be positive",
        ));
    }
    let rows: Vec<Vec<f64>> = (0..measure_from.len())
        .into_par_iter()
        .map(|index| {
            let start = measure_from[index];
            let end = measure_to[index];
            let count = ((end - start) / segment_length).ceil().max(0.0) as usize;
            let mut boundaries: Vec<f64> = (0..count)
                .map(|part| start + part as f64 * segment_length)
                .collect();
            boundaries.push(end);
            boundaries
        })
        .collect();
    let mut values = Vec::new();
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(0_i64);
    for row in rows {
        values.extend(row);
        offsets.push(values.len() as i64);
    }
    Ok((
        PyArray1::from_vec(py, values).unbind(),
        PyArray1::from_vec(py, offsets).unbind(),
    ))
}

fn cumulative_statistic(data: &[f64], q_statistic: bool) -> Vec<f64> {
    if data.len() < 2 {
        return Vec::new();
    }
    let n = data.len() - 1;
    let mut left_sum = vec![0.0; n];
    let mut right_sum = vec![0.0; n];
    let mut left_square_sum = vec![0.0; n];
    let mut right_square_sum = vec![0.0; n];
    let mut sum = 0.0;
    let mut square_sum = 0.0;
    for index in 0..n {
        sum += data[index];
        square_sum += data[index] * data[index];
        left_sum[index] = sum;
        left_square_sum[index] = square_sum;
    }
    sum = 0.0;
    square_sum = 0.0;
    for index in (0..n).rev() {
        let data_index = index + 1;
        sum += data[data_index];
        square_sum += data[data_index] * data[data_index];
        right_sum[index] = sum;
        right_square_sum[index] = square_sum;
    }
    if q_statistic {
        let total_sum: f64 = data.iter().sum();
        let total_square_sum: f64 = data.iter().map(|value| value * value).sum();
        let denominator = total_square_sum - total_sum * total_sum / data.len() as f64;
        (0..n)
            .map(|index| {
                let left_n = (index + 1) as f64;
                let right_n = (n - index) as f64;
                1.0 - ((left_square_sum[index] - left_sum[index] * left_sum[index] / left_n)
                    + (right_square_sum[index] - right_sum[index] * right_sum[index] / right_n))
                    / denominator
            })
            .collect()
    } else {
        (0..n)
            .map(|index| {
                let left_n = (index + 1) as f64;
                let right_n = (n - index) as f64;
                let left =
                    ((left_n * left_square_sum[index] / (left_sum[index] * left_sum[index]) - 1.0)
                        * left_n
                        / (left_n - 1.0))
                        .sqrt();
                let right = ((right_n * right_square_sum[index]
                    / (right_sum[index] * right_sum[index])
                    - 1.0)
                    * right_n
                    / (right_n - 1.0))
                    .sqrt();
                (left + right) / 2.0
            })
            .collect()
    }
}

#[pyfunction]
fn optimal_bisections_pq<'py>(
    py: Python<'py>,
    variables: PyReadonlyArray2<'py, f64>,
    k_mask: PyReadonlyArray1<'py, u8>,
    statistic: i64,
    goal: i64,
) -> PyResult<Py<PyArray1<i64>>> {
    let variables_shape = variables.shape().to_vec();
    let variables = variables.as_slice()?;
    let k_mask = k_mask.as_slice()?;
    if variables_shape.len() != 2 || variables_shape[1] < 2 || k_mask.len() != variables_shape[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "invalid optimal bisection batch shapes",
        ));
    }
    let n_variables = variables_shape[0];
    let n_values = variables_shape[1];
    let k: Vec<usize> = k_mask
        .iter()
        .enumerate()
        .filter_map(|(index, &valid)| if valid != 0 { Some(index) } else { None })
        .collect();
    if k.is_empty() {
        return Ok(PyArray1::from_vec(py, Vec::new()).unbind());
    }
    let statistics: Vec<Vec<f64>> = (0..n_variables)
        .into_par_iter()
        .map(|variable| {
            cumulative_statistic(
                &variables[variable * n_values..(variable + 1) * n_values],
                statistic == 1,
            )
        })
        .collect();
    let mut objective = Vec::with_capacity(k.len().saturating_sub(1));
    for &index in k.iter().filter(|&&index| index > 0) {
        let mut total = 0.0;
        for values in &statistics {
            total += values[index - 1];
        }
        objective.push(total / n_variables as f64);
    }
    let best = if goal == 0 {
        objective.iter().copied().fold(f64::INFINITY, f64::min)
    } else {
        objective.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    };
    let result: Vec<i64> = objective
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| {
            if value == best {
                Some((index + k[0]) as i64)
            } else {
                None
            }
        })
        .collect();
    Ok(PyArray1::from_vec(py, result).unbind())
}

fn segment_ids_impl(
    category_boundaries: &[u8],
    measure_from: &[f64],
    measure_to: &[f64],
    second_from: Option<&[f64]>,
    second_to: Option<&[f64]>,
) -> PyResult<Vec<i64>> {
    let count = category_boundaries.len();
    if measure_from.len() != count
        || measure_to.len() != count
        || second_from.is_some_and(|values| values.len() != count)
        || second_to.is_some_and(|values| values.len() != count)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "segment boundary arrays must have matching lengths",
        ));
    }
    let mut segment_ids = Vec::with_capacity(count);
    let mut segment_id = -1_i64;
    for index in 0..count {
        let starts_new_segment = index == 0
            || category_boundaries[index] != 0
            || (index > 0
                && rounded_thousand(measure_to[index - 1])
                    != rounded_thousand(measure_from[index]))
            || (index > 0
                && second_from.is_some_and(|values| {
                    rounded_thousand(second_to.expect("matching second_to")[index - 1])
                        != rounded_thousand(values[index])
                }));
        if starts_new_segment {
            segment_id += 1;
        }
        segment_ids.push(segment_id);
    }
    Ok(segment_ids)
}

fn rounded_thousand(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

#[pyfunction]
fn segment_ids_by_discontinuity<'py>(
    py: Python<'py>,
    category_boundaries: PyReadonlyArray1<'py, u8>,
    measure_from: PyReadonlyArray1<'py, f64>,
    measure_to: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<i64>>> {
    let category_boundaries = category_boundaries.as_slice()?;
    let measure_from = measure_from.as_slice()?;
    let measure_to = measure_to.as_slice()?;
    let values = if category_boundaries.is_empty() {
        Vec::new()
    } else {
        segment_ids_impl(category_boundaries, measure_from, measure_to, None, None)?
    };
    Ok(PyArray1::from_vec(py, values).unbind())
}

#[pyfunction]
fn segment_ids_by_true_discontinuity<'py>(
    py: Python<'py>,
    category_boundaries: PyReadonlyArray1<'py, u8>,
    measure_from: PyReadonlyArray1<'py, f64>,
    measure_to: PyReadonlyArray1<'py, f64>,
    true_from: PyReadonlyArray1<'py, f64>,
    true_to: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<i64>>> {
    let category_boundaries = category_boundaries.as_slice()?;
    let measure_from = measure_from.as_slice()?;
    let measure_to = measure_to.as_slice()?;
    let true_from = true_from.as_slice()?;
    let true_to = true_to.as_slice()?;
    let values = if category_boundaries.is_empty() {
        Vec::new()
    } else {
        segment_ids_impl(
            category_boundaries,
            measure_from,
            measure_to,
            Some(true_from),
            Some(true_to),
        )?
    };
    Ok(PyArray1::from_vec(py, values).unbind())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_overlapping_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(find_overlapping_intervals_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_all_targets_numeric, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_all_targets_numeric_batch, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_keep_longest_categorical, m)?)?;
    m.add_function(wrap_pyfunction!(linspace_steps_batch, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_bisections_pq, m)?)?;
    m.add_function(wrap_pyfunction!(segment_ids_by_discontinuity, m)?)?;
    m.add_function(wrap_pyfunction!(segment_ids_by_true_discontinuity, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_segment_boundaries_batch, m)?)?;

    m.add_function(wrap_pyfunction!(cumulative_p, m)?)?;
    m.add_function(wrap_pyfunction!(cumulative_q, m)?)?;
    m.add_function(wrap_pyfunction!(longest_overlap_positions, m)?)?;
    m.add_function(wrap_pyfunction!(longest_overlap_positions_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(overlay_events, m)?)?;
    m.add_function(wrap_pyfunction!(cross_sections, m)?)?;
    Ok(())
}
