import numpy as np

from pyroads._backend import announce_fallback

try:
    import pyroads._native as _rust_native
except ImportError:
    _rust_native = None


def linspace_steps_batch(
        measure_from: np.ndarray,
        measure_to: np.ndarray,
        multiples: float,
        minimum_length: float = 0.0,
    ) -> list[np.ndarray]:
    """Expand independent intervals, using the native batch path when enabled."""
    measure_from = np.asarray(measure_from, dtype=np.float64)
    measure_to = np.asarray(measure_to, dtype=np.float64)
    if measure_from.shape != measure_to.shape:
        raise ValueError("measure_from and measure_to must have matching lengths")
    if np.any(measure_from > measure_to):
        raise ValueError("measure from must be less than measure to")

    if _rust_native is not None:
        values, offsets = _rust_native.linspace_steps_batch(
            np.ascontiguousarray(measure_from),
            np.ascontiguousarray(measure_to),
            np.full(measure_from.shape, multiples, dtype=np.float64),
            np.full(measure_from.shape, minimum_length, dtype=np.float64),
        )
        values = np.asarray(values)
        offsets = np.asarray(offsets)
        return [values[start:end] for start, end in zip(offsets[:-1], offsets[1:])]

    announce_fallback()
    return [
        linspace_steps(start, end, multiples, minimum_length)
        for start, end in zip(measure_from, measure_to)
    ]


def fixed_segment_boundaries_batch(
        measure_from: np.ndarray,
        measure_to: np.ndarray,
        segment_length: float,
    ) -> list[np.ndarray]:
    """Return fixed-size interval boundaries for independent numeric ranges."""
    measure_from = np.asarray(measure_from, dtype=np.float64)
    measure_to = np.asarray(measure_to, dtype=np.float64)
    if measure_from.shape != measure_to.shape:
        raise ValueError("measure_from and measure_to must have matching lengths")
    if np.any(measure_from > measure_to):
        raise ValueError("measure from must be less than measure to")
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")

    if _rust_native is not None:
        values, offsets = _rust_native.fixed_segment_boundaries_batch(
            np.ascontiguousarray(measure_from),
            np.ascontiguousarray(measure_to),
            segment_length,
        )
        values = np.asarray(values)
        offsets = np.asarray(offsets)
        return [values[start:end] for start, end in zip(offsets[:-1], offsets[1:])]

    announce_fallback()
    return [
        np.r_[start + np.arange(np.ceil((end - start) / segment_length)) * segment_length, end]
        for start, end in zip(measure_from, measure_to)
    ]


def linspace_steps(measure_from: float, measure_to: float, multiples: float, minimum_length:float=0.0) -> np.ndarray:
    """
    This function is similar to the numpy.linspace function except the list returned
    is;

    - guaranteed to start at `measure_from`,
    - guaranteed to end at `measure_to`,
    - guaranteed to have at least `minimum_length` between each value, and
        - except when `measure_from` and `measure_to` are < `minimum_length`, or if `measure_from` and `measure_to` are the same.
        - no check is performed to ensure that `measure_from` < `measure_to`. Will produce garbage output if this is the case.
    - **mostly** aligned to integer multiples of `multiples`

    The `minimum_length` parameter can cause the first and last segment to be
    Combined with the second or second-last segment respectively

    Args:
        measure_from (float): The starting point of the list
        measure_to (float): The ending point of the list
        multiples (float): Align items of the list to integer multiples of this value
        minimum_length (float, optional): Optionally merge the first and last segment with the second or second-last segment respectively if they would be less than this length. Zero by default.

    Example:

    ```python
    result = linspace_nice_steps(
        measure_from = 190,
        measure_to   = 270.05,
        multiples    = 50
        minimum_length = 0.1
    )
    assert result == [190, 200, 250, 270.05]
    ```


    """
    if measure_from > measure_to:
        raise ValueError("measure from must be less than measure to")

    if _rust_native is not None:
        values, offsets = _rust_native.linspace_steps_batch(
            np.array([measure_from], dtype=np.float64),
            np.array([measure_to], dtype=np.float64),
            np.array([multiples], dtype=np.float64),
            np.array([minimum_length], dtype=np.float64),
        )
        return np.asarray(values)[np.asarray(offsets[0]):np.asarray(offsets[1])]

    announce_fallback()
    left  = np.ceil (measure_from / multiples)
    right = np.floor(measure_to   / multiples)
    num   = right - left

    result = (np.arange(0, num + 1) + left) * multiples

    if len(result) == 0:
        return np.array([measure_from, measure_to])
    else:
        if result[0] != measure_from:
            if np.round(result[0] - measure_from, 6) < minimum_length:
                result[0] = measure_from
            else:
                result = np.append([measure_from], result)
        if result[-1] != measure_to:
            if np.round(measure_to - result[-1], 6) < minimum_length:
                result[-1] = measure_to
            else:
                result = np.append(result, [measure_to])
    return result
