def normalize_points(points):
    mean = np.mean(points, axis=0)
    scale = np.sqrt(2) / np.mean(np.linalg.norm(points - mean, axis=1))

    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0,     0,              1]
    ])

    points_homog = np.hstack([points, np.ones((points.shape[0], 1))])  # Nx3
    points_norm_homog = (T @ points_homog.T).T  # Nx3

    # Convertimos de homogéneas a cartesianas (dividir entre la tercera coordenada)
    points_norm = points_norm_homog[:, :2] / points_norm_homog[:, 2][:, np.newaxis]

    return points_norm, T
