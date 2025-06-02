ef ransac_fundamental_matrix(x1, x2, threshold=1.0, max_iters=2000): #
    best_F = None
    best_inliers = None
    max_inliers_count = 0
    
    for i in range(max_iters):
        # 8 puntos random, f provisional
        indices = np.random.choice(len(x1), 8, replace=False)
        x1_subset = x1[indices]
        x2_subset = x2[indices]      
    
        x1_normalized, T1 = normalize_points(x1_subset)
        x2_normalized, T2 = normalize_points(x2_subset)
        F_normalized = eight_point_algorithm(x1_normalized, x2_normalized)#calculo de F con los 8 puntos normalizados 
        
        # Desnormalizar f
        F = T2.T @ F_normalized @ T1
        
        errors = []
        for i in range(len(x1)):
            x1i = np.append(x1[i], 1)  # Convertir a coordenadas homogéneas
            x2i = np.append(x2[i], 1)  
            error = np.abs(x2i.T @ F @ x1i)# distancia entre el punto al punto epipolar error geometrico
            errors.append(error)#error epipolar cuanto viola la condicion de F 
        inliers = np.array(errors) < threshold #puntos por debajo del umbral
        inliers_count = np.sum(inliers)
        
        # Actualizar el mejor modelo si se encuentra más inliers
        if inliers_count > max_inliers_count:
            best_F = F
            best_inliers = inliers
            max_inliers_count = inliers_count
    
    return best_F, best_inliers #mejor F, y un booleano con los puntos que se cogen
