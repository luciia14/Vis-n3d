import cv2
import matplotlib.pyplot as plt
import numpy as np
def process_and_show_matches_with_epilines(img1_path, img2_path, F, pts1, pts2):

    # Cargar imágenes en escala de grises
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  matchColor=(0, 255, 255), singlePointColor=(0, 0, 255), flags=2)

    plt.figure(figsize=(14, 6))
    plt.imshow(img_matches)
    plt.title("Correspondencias SIFT")
    plt.axis('off')
    plt.show()

    # Calcular líneas epipolares para puntos dados
    lines1 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    def draw_lines(img, lines, pts):
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape
        for r, pt in zip(lines, pts):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = 0, int(-r[2] / r[1])
            x1, y1 = w, int(-(r[2] + r[0] * w) / r[1])
            cv2.line(img_color, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img_color, tuple(pt.astype(int)), 5, color, -1)
        return img_color

    img1_lines = draw_lines(img1, lines2, pts1)
    img2_lines = draw_lines(img2, lines1, pts2)

    # Dibujar epipolos
    epipole1 = np.linalg.svd(F)[2][-1]
    epipole2 = np.linalg.svd(F.T)[2][-1]
    epipole1 = (epipole1 / epipole1[2])[:2].astype(int)
    epipole2 = (epipole2 / epipole2[2])[:2].astype(int)

    cv2.circle(img1_lines, tuple(epipole1), 8, (0, 255, 0), -1)
    cv2.circle(img2_lines, tuple(epipole2), 8, (0, 255, 0), -1)

    # Mostrar imágenes con líneas epipolares y epipolos
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1_lines, cv2.COLOR_BGR2RGB))
    plt.title("Imagen 1 con líneas epipolares y epipolo")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2_lines, cv2.COLOR_BGR2RGB))
    plt.title("Imagen 2 con líneas epipolares y epipolo")
    plt.axis('off')
    plt.show()
