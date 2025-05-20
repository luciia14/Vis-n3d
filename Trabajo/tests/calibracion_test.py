import numpy as np
import pytest
from scripts.zhang_calibration import zhang_calibration

def is_rotation_matrix(R):
    should_be_identity = np.dot(R, R.T)
    I = np.identity(3)
    return np.allclose(should_be_identity, I, atol=1e-6) and np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

def decompose_projection_matrix_lstsq(P):
    M = P[:, :3]
    p4 = P[:, 3]

    Minv = np.linalg.inv(M)
    Q, R = np.linalg.qr(Minv)

    diag_sign = np.sign(np.diag(R))
    D = np.diag(diag_sign)

    Q = Q @ D
    R = D @ R

    K = np.linalg.inv(R)
    K = K / K[2, 2]

    R = Q.T

    t = np.linalg.inv(K) @ p4

    return K, R, t

def test_zhang_calibration_properties():
    # Datos simulados (3 planos con puntos)
    object_points_list = []
    image_points_list = []

    # Generar puntos 3D (planos Z=0) y puntos 2D simulados (ejemplo sencillo)
    for i in range(3):
        obj_pts = []
        img_pts = []
        for x in range(4):
            for y in range(3):
                X, Y, Z = x*0.03, y*0.03, 0  # plano en Z=0 con separación 3cm
                obj_pts.append([X, Y, Z])

                # Simulación imagen (píxeles) con una proyección simple
                u = 500 + 100 * X - 50 * Y + np.random.normal(0, 0.5)
                v = 400 + 80 * Y + 30 * X + np.random.normal(0, 0.5)
                img_pts.append([u, v])
        object_points_list.append(np.array(obj_pts))
        image_points_list.append(np.array(img_pts))

    # Ejecutamos la calibración
    P = zhang_calibration(object_points_list, image_points_list)

    # Comprobaciones básicas de P
    assert P.shape == (3, 4), "La matriz P debe ser 3x4"

    # Comprobar normalización (último elemento = 1)
    assert np.isclose(P[2, 3], 1), "P no está normalizada correctamente"

    # Comprobar rango
    assert np.linalg.matrix_rank(P) == 3, "La matriz P debe tener rango 3"

    # Verificar correspondencia de puntos (error reproyección pequeño)
    total_error = 0
    total_points = 0
    for i in range(len(image_points_list)):
        obj_pts = object_points_list[i]
        img_pts = image_points_list[i]
        for j in range(len(obj_pts)):
            X, Y, Z = obj_pts[j]
            homog_obj = np.array([X, Y, Z, 1])
            proj = P @ homog_obj
            proj = proj / proj[2]
            u_proj, v_proj = proj[0], proj[1]
            u_real, v_real = img_pts[j]
            error = np.sqrt((u_proj - u_real)**2 + (v_proj - v_real)**2)
            total_error += error
            total_points += 1

    mean_error = total_error / total_points
    assert mean_error < 2, f"Error de reproyección demasiado alto: {mean_error:.2f} píxeles"

    # Descomponer P para comprobar K,R,t
    K, R, t = decompose_projection_matrix_lstsq(P)

    # Validar matriz intrínseca K
    assert K.shape == (3, 3), "K debe ser 3x3"
    assert np.isclose(K[2, 2], 1), "Elemento K[2,2] debe ser 1"
    assert K[0, 0] > 0 and K[1, 1] > 0, "Focales deben ser positivas"
    assert abs(K[0, 1]) < 1e-3, "Skew debería ser cercano a cero"

    # Validar matriz rotación R
    assert is_rotation_matrix(R), "R no es matriz de rotación válida"

    # Validar vector traslación t es 3x1
    assert t.shape == (3,), "t debe ser vector de 3 elementos"


if __name__ == "__main__":
    pytest.main()
