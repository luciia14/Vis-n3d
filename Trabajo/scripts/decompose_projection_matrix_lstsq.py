def decompose_projection_matrix_lstsq(P):
  
    M = P[:, :3]  
    t = P[:, 3]  #ultima columna
    
    # Descomposición QR de la inversa de M
    Q, U = np.linalg.qr(np.linalg.inv(M))
    
 
    D = np.diag(np.sign(np.diag(U)) * [-1, -1, 1])  # Ajustamos los signos de U
    Q = np.dot(Q, D)  # Aplicamos la matriz D a Q
    U = np.dot(D, U)  # Aplicamos la matriz D a U
    
    #  K (distancia focal)
    K = np.linalg.inv(U / U[2, 2])  # La matriz K se obtiene con la inversión de U normalizada
    
    #  rotación R
    s = np.linalg.det(Q)  # Determinante de Q, para verificar el signo de la rotación
    R = s * Q.T  # La rotación se obtiene de Q transpuesta ajustada por el determinante
    
    #  vector de traslación t
    t = s * np.dot(U, P[:, 3])  # El vector de traslación se calcula a partir de U y P
    
    return K, R, t
