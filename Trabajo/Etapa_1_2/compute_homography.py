def compute_homography(X, x):
    X_norm, T_X = normalize_points(X)
    x_norm, T_x = normalize_points(x)
    A = []
    for i in range(X.shape[0]):
        X_i, Y_i = X_norm[i]
        x_i, y_i = x_norm[i]
        A.append([-X_i, -Y_i, -1, 0, 0, 0, x_i*X_i, x_i*Y_i, x_i])
        A.append([0, 0, 0, -X_i, -Y_i, -1, y_i*X_i, y_i*Y_i, y_i])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    # Desnormalizar la homografía
    H = np.linalg.inv(T_x) @ H_norm @ T_X
    return H / H[-1, -1]  # Normalizar para que H[2,2] = 1

# Cálculo de homografías
homographies = []
for obj, img in zip(obj_points, img_points):
    H = compute_homography(obj[:, :2], img)
    homographies.append(H)
