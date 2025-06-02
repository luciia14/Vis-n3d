def decompose_projection_matrix_lstsq(P):
    # Extraemos M (3x3) y el vector de traslación p4
    M = P[:, :3]
    p4 = P[:, 3]

    # Invertimos M para poder hacer la descomposición QR
    Minv = np.linalg.inv(M)

    # QR de la inversa de M
    Q, R = np.linalg.qr(Minv)

    # Ajustamos signos para que la diagonal de R sea positiva
    diag_sign = np.sign(np.diag(R))
    D = np.diag(diag_sign)

    Q = Q @ D
    R = D @ R

    # La matriz intrínseca K es la inversa de R (normalizada)
    K = np.linalg.inv(R)
    K = K / K[2, 2]  # Normalizamos para que K[2,2] = 1

    # La matriz de rotación R es Q transpuesta
    R = Q.T

    # Vector de traslación t: t = K^{-1} * p4
    t = np.linalg.inv(K) @ p4

    return K, R, t
