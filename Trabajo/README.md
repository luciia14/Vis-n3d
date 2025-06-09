# Visión Estereoscópica y Reconstrucción 3D

Este proyecto implementa un pipeline completo de visión estereoscópica para la reconstrucción 3D utilizando Python y NumPy. El sistema sigue las principales etapas necesarias para la reconstrucción tridimensional a partir de imágenes estéreo, permitiendo la estimación de la geometría 3D de una escena utilizando cámaras calibradas o no calibradas.

## Etapas del Pipeline

1. **Calibración de la Cámara**  
   Se emplea el método de calibración de Zhang para estimar los parámetros intrínsecos y extrínsecos de la cámara utilizando imágenes de un patrón de ajedrez.

2. **Factorización de la Matriz de Proyección**  
   Se descompone la matriz de proyección \( P \) en sus componentes: la matriz intrínseca \( K \), la matriz de rotación \( R \) y el vector de traslación \( t \), usando descomposición QR.

3. **Cálculo de la Matriz Fundamental \( F \)**  
   Se detectan y emparejan puntos característicos entre las imágenes usando SIFT y RANSAC, luego se estima la matriz fundamental \( F \) que relaciona los puntos correspondientes entre las dos imágenes.

4. **Cálculo de la Matriz Esencial \( E \)**  
   A partir de la matriz fundamental \( F \) y los parámetros intrínsecos \( K \), se calcula la matriz esencial \( E \), que es utilizada en la triangulación y reconstrucción 3D.

5. **Rectificación Estereoscópica Sin Calibración**  
   Se realiza la rectificación de las imágenes sin necesidad de calibración, utilizando la matriz fundamental \( F \), lo que permite que las líneas epipolares sean horizontales y facilita la búsqueda de correspondencias.

6. **Generación de Nube de Puntos 3D**  
   A partir de la disparidad calculada entre las imágenes rectificadas, se genera una nube de puntos 3D utilizando block matching y un algoritmo de interpolación subpixel.

## Dependencias

- **Python** (versión 3.x)
- **NumPy** para procesamiento numérico.
- **OpenCV** para el manejo de imágenes y funciones de visión por computadora.
- **Matplotlib** para la visualización de resultados.

## Instalación
Instala las dependencias:
pip install -r requirements.txt

## Ejecución
1. Para ejecutar el pipeline completo, usa el siguiente comando:
python main.py
2. Asegúrate de tener las imágenes de entrada en el directorio data/ o ajusta las rutas en el código según sea necesario.

##Resultados
-Matriz Fundamental 𝐹: Calculada y validada con éxito mediante RANSAC.
-Rectificación Estereoscópica:  Tras aplicar la rectificación estéreo sin calibración, las imágenes fueron alineadas de modo que las líneas epipolares son horizontales.
-Nube de Puntos 3D: Generada a partir de la disparidad calculada y visualizada correctamente.

## Conclusión
Este proyecto proporciona una implementación desde cero de un sistema de visión estereoscópica, permitiendo la reconstrucción 3D precisa a partir de imágenes estéreo. A través de los algoritmos de calibración, factorización de matrices, y triangulación, hemos logrado una representación 3D de la escena que puede ser utilizada para una variedad de aplicaciones en visión por computadora.
