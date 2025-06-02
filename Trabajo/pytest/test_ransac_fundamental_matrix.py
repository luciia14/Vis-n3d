# -- coding: utf-8 --
import numpy as np
import pytest
from final import ransac_fundamental_matrix

def test_ransac_fundamental_matrix():
    F_true = np.array([
        [0, -0.0003, 0.06],
        [0.0003, 0, -0.04],
        [-0.05, 0.03, 1]
    ])

    np.random.seed(42)
    num_points = 100
    x1 = np.random.rand(num_points, 2) * 100
    x1_h = np.hstack([x1, np.ones((num_points, 1))])
    lines2 = (F_true @ x1_h.T).T

    x2 = []
    for i in range(num_points):
        a, b, c = lines2[i]
        x, y = x1[i]
        y2 = (-c - a * x) / b if b != 0 else y
        x2.append([x + np.random.normal(0, 0.5), y2 + np.random.normal(0, 0.5)])
    x2 = np.array(x2)

    F_est, inliers = ransac_fundamental_matrix(x1, x2, threshold=1.0)

    assert F_est.shape == (3, 3)
    assert inliers.shape == (num_points,)
    assert np.sum(inliers) > num_points * 0.7

