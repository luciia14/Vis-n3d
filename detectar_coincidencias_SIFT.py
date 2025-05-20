import cv2
import numpy as np
def detectar_coincidencias_SIFT(img1,img2):
    
    # Detectar características SIFT
    sift = cv2.SIFT_create()
    puntos_clave1, descriptores1 = sift.detectAndCompute(img1, None)
    puntos_clave2, descriptores2 = sift.detectAndCompute(img2, None)#puntos clave  lista de objetos con posicion, orientacion
    
    # Emparejar características
    bf = cv2.BFMatcher(cv2.NORM_L2)#matcher con distancia euclidea
    coincidencias = bf.knnMatch(descriptores1, descriptores2, k=2)
    
    # Filtrar coincidencias con la prueba, david lower, matches buenos
    buenas_coincidencias = []
    for m, n in coincidencias:
        if m.distance < 0.5 * n.distance:
            buenas_coincidencias.append(m)
    print(f"Buenas coicidencias {len(buenas_coincidencias)}")
    if len(buenas_coincidencias) < 4:
        print("No hay suficientes coincidencias buenas.")
        exit()
    
    # Extraer puntos correspondientes
    puntos1 = np.float32([puntos_clave1[m.queryIdx].pt for m in buenas_coincidencias]) #cordenadas 2d de los puntos claves 
    puntos2 = np.float32([puntos_clave2[m.trainIdx].pt for m in buenas_coincidencias])
    return puntos1, puntos2, coincidencias
