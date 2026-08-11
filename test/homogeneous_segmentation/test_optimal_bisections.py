from io import StringIO
import pandas as pd
import numpy as np
from pyroads.homogeneous_segmentation._optimal_bisections import optimal_bisections
from pyroads.homogeneous_segmentation._cumulative_q import cumulative_q
from pyroads.homogeneous_segmentation._cumulative_p import cumulative_p

def test_optimal_bisections_max_q_cumulative():
    data = pd.read_csv(StringIO("""road,slk_from,slk_to,cwy,deflection,dirn
    H001,0.00,0.01,L,179.37,L
    H001,0.01,0.02,L,177.12,L
    H001,0.02,0.03,L,179.06,L
    H001,0.03,0.04,L,212.65,L
    H001,0.04,0.05,L,175.35,L
    H001,0.05,0.06,L,188.66,L
    H001,0.06,0.07,L,188.31,L
    H001,0.07,0.08,L,174.48,L
    H001,0.08,0.09,L,210.28,L
    H001,0.09,0.10,L,260.05,L
    H001,0.10,0.11,L,228.83,L
    H001,0.11,0.12,L,226.33,L
    H001,0.12,0.13,L,245.53,L
    H001,0.13,0.14,L,315.77,L
    H001,0.14,0.15,L,373.86,L
    H001,0.15,0.16,L,333.56,L"""))
    # in R the result is 10 but R uses index from 1
    expected_result = 9
    actual_result = optimal_bisections(
        variables                  = [data["deflection"].values],
        length                     = (data["slk_to"] - data["slk_from"]).values,
        minimum_segment_length     = 0.030,
        cumulative_split_statistic = cumulative_q,
        goal                       = "max"
    )
    assert actual_result[0] == expected_result
    assert actual_result.dtype == np.int64


def test_optimal_bisections_min_p_cumulative():
    data = pd.read_csv(StringIO("""road,slk_from,slk_to,cwy,deflection,dirn
    H001,0.00,0.01,L,179.37,L
    H001,0.01,0.02,L,177.12,L
    H001,0.02,0.03,L,179.06,L
    H001,0.03,0.04,L,212.65,L
    H001,0.04,0.05,L,175.35,L
    H001,0.05,0.06,L,188.66,L
    H001,0.06,0.07,L,188.31,L
    H001,0.07,0.08,L,174.48,L
    H001,0.08,0.09,L,210.28,L
    H001,0.09,0.10,L,260.05,L
    H001,0.10,0.11,L,228.83,L
    H001,0.11,0.12,L,226.33,L
    H001,0.12,0.13,L,245.53,L
    H001,0.13,0.14,L,315.77,L
    H001,0.14,0.15,L,373.86,L
    H001,0.15,0.16,L,333.56,L"""))

    expected_result = 9 # result in R is 10 but R uses index from 1
    actual_result = optimal_bisections(
        variables                  = [data["deflection"].values],
        length                     = (data["slk_to"] - data["slk_from"]).values,
        minimum_segment_length     = 0.030,
        cumulative_split_statistic = cumulative_p,
        goal                       = "min"
    )
    assert actual_result[0] == expected_result
    assert actual_result.dtype == np.int64