
import pandas
import numpy as np
import numpy.typing as npt
from typing import Callable, Literal

_goal_functions = {
    "min": np.min,
    "max": np.max
}

def optimal_bisections (
        variables:list[npt.NDArray[np.float64]],
        length:npt.NDArray[np.float64],
        minimum_segment_length:float,
        cumulative_split_statistic:Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        goal:Literal["min", "max"] = "max"
    ) -> npt.NDArray[np.int64]:
    """
    Bisects the given data at either the minimum and maximum of the
    'cumulative_split_statistic' function.

    This function enforces a minimum

    Args:
        data (pandas.DataFrame): The DataFrame containing the data to be segmented.
        var (list[str]): A list of column names in 'data' that are to be used for segmentation.
        length_column (str): The name of the column that represents the length of each segment.
        min_length (float): Minimum allowed lengths for each segment.

    Returns:
        list[int]: A list of indices in 'data' where the maximum Q values are found, indicating optimal split points.
    """

    goal_function = _goal_functions.get(goal)
    if goal_function is None:
        raise ValueError(f"goal must be one of {list(_goal_functions.keys())}")

    cumulative_length_left  = np.cumsum(length)
    cumulative_length_right = np.cumsum(length[::-1])[::-1]
    
    k_mask = ~ (
          (cumulative_length_left  <= minimum_segment_length)
        | (cumulative_length_right <= minimum_segment_length)
    )

    # to match the behaviour of the R script we must expand the false values
    # in the array by one index either side;
    # the bool series 001111000 is modified to 0001100000
    # Since cumlength_left and cumlength_right should be monotonically increasing,
    # the bool series should have at most 3 segments, with
    # contiguous sections of zero only at the start and end of the series.
    k_mask = ~np.convolve(~k_mask, [True, True, True])[1:-1] # TODO: check if segond arg must be coerced to numpy array?
    

    
    # plt.figure()
    # (pandas.Series(k_mask,index=cumlength_left).astype("int") * 0.1).plot(color="grey", marker="x")
    # pandas.Series(cumlength_left , index=cumlength_left).plot(marker=".")
    # pandas.Series(cumlength_right, index=cumlength_left).plot(marker=".")
    # plt.axhline(min_length)


    k = np.flatnonzero(k_mask)
    
    try:
        # confirm that k_mask splits the data into three portions
        # this appears to be assumed in the R package?
        assert len(np.split(k_mask,np.flatnonzero(k_mask[:-1] != k_mask[1:])+1)) == 3
    except AssertionError:
        # if it didn't split into 3 sections, perhaps one or two segments will not 
        # print("did not split into 3... try 1 or 2?")
        assert len(np.split(k_mask,np.flatnonzero(k_mask[:-1] != k_mask[1:])+1)) in {1,2}


    objective_columns = []
    for variable in variables:
        # qvalue[, i] <- _cumq(data.var[[i]])[k - 1]
        objective_columns.append(
            cumulative_split_statistic(variable)[k_mask[1:]]
        )
    
    # qvalue = rowMeans(qvalue)
    mean_objective = np.mean(objective_columns, axis=0)
    
    # NOTE: This next line appears to retrieve the index of the global maximum (mean) Q value.
    #       It IS possible that it could return a list instead of a scalar
    #       The downstream code will split at every index in maxk without complaint.
    #       Thats ok, BUT there is no further check that we do not create a segment shorter
    #       than the minimum segment length.
    #       we could fix this my using np.argmax() which will return only the first maximum.
    #       for now I will leave this as-is so that results are compareable with the R code.
    # maxk <- which(qvalue == max(qvalue)) + max(k_left)
    
    maxk = np.flatnonzero(
        mean_objective == goal_function(mean_objective)
    ) + k[0]
    return maxk
