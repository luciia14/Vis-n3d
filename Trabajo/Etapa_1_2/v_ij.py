def v_ij(H, i, j):
    return np.array([
        H[0,i]*H[0,j],
        H[0,i]*H[1,j] + H[1,i]*H[0,j],
        H[1,i]*H[1,j],
        H[2,i]*H[0,j] + H[0,i]*H[2,j],
        H[2,i]*H[1,j] + H[1,i]*H[2,j],
        H[2,i]*H[2,j]
    ])

V = []
for H in homographies:
    V.append(v_ij(H, 0, 1))
    V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))
V = np.array(V)

_, _, Vt = np.linalg.svd(V)
b = Vt[-1]

# 4. Calcular B y obtener K con Cholesky
B = np.array([
    [b[0], b[1], b[3]],
    [b[1], b[2], b[4]],
    [b[3], b[4], b[5]]
])

if np.trace(B) < 0:
    B = -B

try:
    iK = cholesky(B)
except np.linalg.LinAlgError:
    raise Exception("B no es definida positiva")

K = np.linalg.inv(iK)
K = K / K[2, 2]
print("Matriz K (intrínseca):\n", K)

# 5. Calcular extrínsecos para cada vista
extrinsics = []
projections = []

for H in homographies:
    A = np.linalg.inv(K).dot(H)
    A = A / np.linalg.norm(A[:, 0])  # normalizar

    r1 = A[:, 0]
    r2 = A[:, 1]
    t = A[:, 2]
    r3 = np.cross(r1, r2)

    R_approx = np.stack((r1, r2, r3), axis=1)
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt

    if np.dot([0, 0, 1], -R.T @ t) < 0:
        R[:, :2] = -R[:, :2]
        t = -t

    extrinsics.append((R, t))
    P = K @ np.hstack((R, t.reshape(3,1)))
    projections.append(P)

print("\nParámetros extrínsecos (R, t) por imagen:")
for i, (R, t) in enumerate(extrinsics):
    print(f"\nImagen {i+1}:")
    print("R =\n", R)
    print("t =\n", t)

# Matriz de proyección P = K[R|t]
print("\nMatrices de proyección:")
for i, P in enumerate(projections):
    print(f"\nP[{i+1}] =\n", P) 
