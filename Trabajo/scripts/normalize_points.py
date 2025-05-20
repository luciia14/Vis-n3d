def normalize_points(points):
    
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # Obtener la media y desviación estándar
    mean = np.mean(points_homogeneous, axis=0)
    std_dev = np.std(points_homogeneous)

    # Matriz de normalización
    T = np.array([[1/std_dev, 0, -mean[0]/std_dev],
                  [0, 1/std_dev, -mean[1]/std_dev],
                  [0, 0, 1]])

    # Normalizar los puntos
    points_normalized = np.dot(T, points_homogeneous.T).T

    return points_normalized, T
