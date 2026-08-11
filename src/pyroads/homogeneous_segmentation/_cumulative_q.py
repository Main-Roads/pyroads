"""
This is a private module containing the implementation of the cumulative `Q statistic`
which supports Spatial Heterogeneity-based Segmentation (SHS) method.
"""
import numpy as np
import numpy.typing as npt


def cumulative_q (data:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ Computes the cumulative Q-statistic for each potential split index in an array.

    The Q-statistic is a measure that quantifies the importance of each variable by
    comparing the variance of the target variable across different spatial regions
    against the overall variance in the entire study area. It is used to identify
    optimal segmentation points in the dataset where the variance within segments is
    minimized, and the variance between segments is maximized.

    Formula:

    ```
    Q = 1 - Σ(N_v,j * σ_v,j^2) / (N_v * σ_v^2)
    ```

    Where:

    - `N_v` and `σ_v^2` are the number and population variance of observations within the whole study area.
    - `N_v,j` and `σ_v,j^2` are the number and population variance of observations within the j-th sub-region.
    - `M` is the total number of sub-regions.

    Args:
        data (npt.ArrayLike): An array-like object containing the data points for which the Q values are to be computed.

    Returns:
        npt.ArrayLike: An array of the same length as 'data', containing the computed Q values for each potential split
        point.
    """

    cum_n                = np.arange(1, len(data))  # 1:(n - 1)         # Used as denominator later ∴ Must start from 1.
    # This next line ensures that we have preserved the original functionality; that cum_n is 1 item shorter than data
    # (Possibly it is an error, or maybe it is intended that part of the math )
    assert len(cum_n) == len(data) - 1

    data_left            = data[:-1]     # drops last
    data_right           = data [:0 :-1] # reverses and then drops last

    cum_data_left        = np.cumsum(data_left      )
    cum_data_right       = np.cumsum(data_right     )[::-1]
    sumd                 = np.sum   (data           )
    cum_datasquare_left  = np.cumsum(data_left  ** 2)
    cum_datasquare_right = np.cumsum(data_right ** 2)[::-1]
    #with np.errstate(invalid='ignore', divide='ignore'):
    result = (
        1 - (
            ( cum_datasquare_left - cum_data_left  * cum_data_left  / cum_n      ) 
            + (cum_datasquare_right - cum_data_right * cum_data_right / cum_n[::-1])
        ) / (
            np.sum(np.power(data, 2)) - sumd**2.0 / len(data)
        )
    )
    return result

