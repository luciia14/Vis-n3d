def eight_point_algorithm(x1, x2):#calcula la matriz f a partir de 8 pares de puntos corresponidentes 
    
    A = []
    for i in range(8):
        x1i, y1i = x1[i][0], x1[i][1]
        x2i, y2i = x2[i][0], x2[i][1]
        A.append([x2i * x1i, x2i * y1i, x2i, y2i * x1i, y2i * y1i, y2i, x1i, y1i, 1])
    
    A = np.array(A)  
    #svd
    _, _, Vt = np.linalg.svd(A)
    
    # coge el ultimo vector 
    F = Vt[-1].reshape(3, 3)
    
    # para que sea rango 2
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0 #valor singular 0 ultimo
    F = np.dot(U, np.dot(np.diag(S), Vt))
    
    return F
