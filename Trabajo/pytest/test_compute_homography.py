# -- coding: utf-8 --
import numpy as np
import pytest
from final import compute_homography  

def test_compute_homography():
    # Puntos en el plano del objeto (cuadrado unitario)
    X = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])

    # Puntos transformados en imagen mediante una homografía conocida
    true_H = np.array([
        [2, 0.2, 100],
        [0.1, 1.5, 50],
        [0.001, 0.002, 1]
    ])
    X_h = np.hstack([X, np.ones((4, 1))])  # Puntos homogéneos
    x_h = (true_H @ X_h.T).T
    x = x_h[:, :2] / x_h[:, 2][:, np.newaxis]  # Convertimos a coordenadas cartesianas

    # Calculamos la homografía con la función
    H_est = compute_homography(X, x)

    # Aplicamos H_est a X y comparamos con x
    x_est_h = (H_est @ X_h.T).T
    x_est = x_est_h[:, :2] / x_est_h[:, 2][:, np.newaxis]

    # Comprobamos que los puntos transformados estén muy cerca de los esperados
    assert np.allclose(x_est, x, atol=1e-5)

    # Verificamos que la homografía esté normalizada
    assert np.isclose(H_est[2, 2], 1.0, atol=1e-6)

