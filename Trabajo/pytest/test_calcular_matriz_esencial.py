# -- coding: utf-8 --
import numpy as np
import pytest
from final import calcular_matriz_esencial

def test_calcular_matriz_esencial():
    F = np.array([
        [1e-6, 2e-6, -3e-4],
        [-2e-6, 1e-6, 4e-4],
        [3e-4, -4e-4, 1]
    ])

    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])

    E = calcular_matriz_esencial(F, K)

    assert E.shape == (3, 3)
    assert not np.allclose(E, np.zeros((3, 3)))
    assert np.linalg.matrix_rank(E) == 3

