from pyroads.homogeneous_segmentation._cumulative_q import cumulative_q
import numpy as np

def test_q_cumulative():
    """Test against output from R program to confirm identical"""
    input = np.array([
        179.37,177.12,179.06,212.65,175.35,188.66,188.31,174.48,210.28,
        260.05,228.83,226.33,245.53,315.77,373.86,333.56
    ])
    r_output = np.array([
        0.04582494, 0.10266886, 0.16409027, 0.16409040, 0.24921584, 0.31932834,
        0.40606724, 0.55653928, 0.62684644, 0.55570908, 0.60795648, 0.70864689,
        0.79360021, 0.60877663, 0.19950557,
    ])
    python_output = cumulative_q(input)
    assert np.allclose(python_output, r_output, atol=1e-8)
