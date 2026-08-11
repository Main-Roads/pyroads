import numpy as np


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
