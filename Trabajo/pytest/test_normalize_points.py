# -- coding: utf-8 --
import numpy as np
import pytest
from final import normalize_points

def test_normalize_points():
    # Creamos un conjunto de puntos de prueba
    points = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])

    # Ejecutamos la función
    norm_points, T = normalize_points(points)

    # Verificamos la forma de los resultados
    assert norm_points.shape == points.shape
    assert T.shape == (3, 3)

    # Verificamos que la media esté (aproximadamente) en el origen
    mean = np.mean(norm_points, axis=0)
    assert np.allclose(mean, [0, 0], atol=1e-6)

    # Verificamos que la media de las distancias al origen sea sqrt(2)
    distances = np.linalg.norm(norm_points, axis=1)
    mean_distance = np.mean(distances)
    assert np.isclose(mean_distance, np.sqrt(2), atol=1e-6)

