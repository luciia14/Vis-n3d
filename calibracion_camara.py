
# Cargar las imágenes corregidas del tablero de ajedrez que hemos hecho previamente
images = glob.glob('./undistorted_*.jpeg')

# Definimos los parámetros del tablero para la detección
square_size = 25  # Tamaño de cada cuadro
pattern_size = (7, 7)  # Número de cuadros del patrón de ajedrez
contador = 0

# Crear las coordenadas 3D reales de los puntos del tablero
object_points = np.zeros((np.prod(pattern_size), 3), np.float32) 

# Suponemos que el patrón es plano sobre z=0, y que cada cuadro tiene tamaño "square_size"
# Se genera una lista con las posiciones (x, y, 0) de cada esquina, organizadas en filas y columnas
object_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) * square_size # 

# Listas para almacenar los puntos 3D y 2D
obj_points = []  # 3D objeto real el tablero
img_points = []   # 2D la imagen 

# Leer la primera imagen para obtener el tamaño
img = cv2.imread(images[0])
h, w = img.shape[:2]

# Definir parámetros de la cámara
camera_matrix = np.array([[w, 0, w / 2],
                          [0, w, h / 2],
                          [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros(4)  # distorsion 0 


for image_path in images:
    img = cv2.imread(image_path)
    
    # Convierte la imagen a escal de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # Detectar las esquinas
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret: #si detecta las esquinas las añade en la lista
        obj_points.append(object_points)  # añade los ptos 3d
        img_points.append(corners)  # Agregar ptos 2d
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)# dibuja esquinas
        cv2.imshow('Imagen de ajedrez', img) #muestra imagen
        cv2.waitKey(500)  
    else: #por is no detecta las esquinas
        print("No se han encontrado esquinas",1 )

for i, (obj, img) in enumerate(zip(obj_points, img_points)):
    print(f"Imagen {i+1}: {len(obj)} puntos 3D - {len(img)} puntos 2D")

print(f"\nTotal de imágenes válidas: {len(obj_points)}")
print(f"Total de correspondencias: {len(obj_points) * len(object_points)}")
cv2.destroyAllWindows()

