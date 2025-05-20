[19:30, 20/05/2025] Alba cucala: import numpy as np
import pytest

@pytest.fixture
def matriz_essencial():
    # Matriz esencial ejemplo (3x3)
    E = np.array([
        [0, -0.2, 0.3],
        [0.2, 0, -0.5],
        [-0.3, 0.5, 0]
    ])
    return E

@pytest.fixture
def puntos_2d():
    # Puntos 2D homogéneos en la primera imagen (x) y la segunda (x')
    x = np.array([0.5, 0.3, 1.0])    # ejemplo punto en cámara 1
    xp = np.array([0.4, 0.35, 1.0])  # ejemplo punto en cámara 2
    return x, xp

@pytest.fixture
def epipolos(matriz_essencial):
    E = matriz_essencial
    # Calcular epípolos como vectores propios de E y E^T asociados a valor propio 0
    # Se calcula el kernel (nullspace) de E y E.T
    # Usamos SVD para encontrar el vector singular asociado al menor valor singular
    U…
[19:31, 20/05/2025] Alba cucala: import numpy as np
import pytest

@pytest.fixture
def matriz_essencial():
    # Matriz esencial ejemplo
    E = np.array([
        [0, -0.2, 0.3],
        [0.2, 0, -0.5],
        [-0.3, 0.5, 0]
    ])
    return E

@pytest.fixture
def puntos_imagen():
    # Puntos 2D homogéneos en la imagen (coordenadas de pixel o normalizadas)
    x = np.array([500, 300, 1.0])    # punto en imagen 1
    xp = np.array([520, 310, 1.0])   # punto correspondiente en imagen 2
    return x, xp

@pytest.fixture
def epipolos(matriz_essencial):
    E = matriz_essencial
    U, S, Vt = np.linalg.svd(E)
    e = Vt[-1]
    U2, S2, Vt2 = np.linalg.svd(E.T)
    ep = Vt2[-1]
    return e / e[-1], ep / ep[-1]

def test_longuet_higgins(matriz_essencial, puntos_imagen):
    E = matriz_essencial
    x, xp = puntos_imagen
    val = xp.T @ E @ x
    assert np.isclose(val, 0, atol=1e-4), f"Longuet-Higgins fallo: {val}"

def test_lineas_epipolares(matriz_essencial, puntos_imagen):
    E = matriz_essencial
    x, xp = puntos_imagen

    l_prime = E @ x
    l = E.T @ xp

    val1 = x.T @ l
    val2 = xp.T @ l_prime

    assert np.isclose(val1, 0, atol=1e-4), f"Línea epipolar l falla: {val1}"
    assert np.isclose(val2, 0, atol=1e-4), f"Línea epipolar l' falla: {val2}"

def test_epipolos(matriz_essencial, epipolos):
    E = matriz_essencial
    e, ep = epipolos

    val1 = e.T @ E
    val2 = E @ ep

    assert np.allclose(val1, 0, atol=1e-4), f"Epípolo e falla: {val1}"
    assert np.allclose(val2, 0, atol=1e-4), f"Epípolo e' falla: {val2}"
