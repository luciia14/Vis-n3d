# -- coding: utf-8 --
import numpy as np
import pytest
from final import eight_point_algorithm

def test_eight_point_algorithm():
    F_true = np.array([
        [0, -0.0003, 0.06],
        [0.0003, 0, -0.04],
        [-0.05, 0.03, 1]
    ])

    np.random.seed(0)
    x1 = np.random.rand(8, 2) * 100
    x1_h = np.hstack([x1, np.ones((8, 1))])

    x2_h = (F_true @ x1_h.T).T
    x2 = x2_h[:, :2] / x2_h[:, 2][:, np.newaxis]

    F_est = eight_point_algorithm(x1, x2)

    # Verificamos que x2.T * F * x1 ≈ 0
    for i in range(8):
        x1_h_i = np.append(x1[i], 1)
        x2_h_i = np.append(x2[i], 1)
        val = x2_h_i @ F_est @ x1_h_i
        assert np.isclose(val, 0, atol=1e-3)

