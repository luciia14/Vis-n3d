# -- coding: utf-8 --
import numpy as np
import pytest
from final import decompose_projection_matrix_lstsq

def test_decompose_projection_matrix_lstsq():
    K = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ])
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    t = np.array([10, 20, 30])

    Rt = np.hstack([R, t.reshape(-1, 1)])
    P = K @ Rt

    K_est, R_est, t_est = decompose_projection_matrix_lstsq(P)

    assert np.allclose(K_est, K, atol=1e-5)
    assert np.allclose(R_est @ R_est.T, np.eye(3), atol=1e-5)
    assert np.linalg.det(R_est) > 0
    assert np.allclose(P[:, 3] / P[2, 3], K_est @ t_est / K_est[2, 2], atol=1e-5)

