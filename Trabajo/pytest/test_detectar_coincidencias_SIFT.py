# -- coding: utf-8 --
import numpy as np
import cv2
import pytest
from final import detectar_coincidencias_SIFT

def test_detectar_coincidencias_SIFT():
    # Creamos dos imágenes sintéticas simples
    img1 = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img1, (100, 100), 40, 255, -1)

    # Segunda imagen: igual pero con pequeña traslación
    M = np.float32([[1, 0, 5], [0, 1, 10]])
    img2 = cv2.warpAffine(img1, M, (200, 200))

    puntos1, puntos2, coincidencias = detectar_coincidencias_SIFT(img1, img2)

    assert len(puntos1) >= 4
    assert puntos1.shape == puntos2.shape
    assert puntos1.shape[1] == 2

