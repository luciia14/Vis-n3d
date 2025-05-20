from normalize_points import normalize_points
from eight_point_algorithm import eight_point_algorithm
import numpy as np
def ransac_fundamental_matrix(x1, x2, threshold=0.5, max_iters=2000): #
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
            l2 = F @ x1i
            l1 = F.T @ x2i
            
            dist1 = np.abs(x2i.T @ F @ x1i) / np.sqrt(l2[0]**2 + l2[1]**2)
            dist2 = np.abs(x2i.T @ F @ x1i) / np.sqrt(l1[0]**2 + l1[1]**2)
            
            error_geom = dist1 + dist2
            errors.append(error_geom)
        
        errors = np.array(errors)
        inliers = errors < threshold
        inliers_count = np.sum(inliers)
        
        # Actualizar el mejor modelo si se encuentra más inliers
        if inliers_count > max_inliers_count:
            best_F = F
            best_inliers = inliers
            max_inliers_count = inliers_count
    
    return best_F, best_inliers #mejor F, y un booleano con los puntos que se cogen
