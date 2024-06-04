import cv2
import numpy as np
import matplotlib.pyplot as plt

# img_rutas = {'top': 'imagenes/SENOS1-T.BMP', 'bottom': 'imagenes/SENOS1-B.BMP', 'left': 'imagenes/SENOS1-L.BMP',
#              'right': 'imagenes/SENOS1-R.BMP', 'textura': 'imagenes/SENOS1-S.BMP'}

# i_1 = cv2.imread('imagenes/SENOS1-T.BMP', cv2.IMREAD_GRAYSCALE)
i_1 = cv2.imread('imagenes/SENOS1-T.BMP', cv2.IMREAD_UNCHANGED)

# i_2 = cv2.imread('imagenes/SENOS1-R.BMP',cv2.IMREAD_GRAYSCALE)
# i_3 = cv2.imread('imagenes/SENOS1-B.BMP',cv2.IMREAD_GRAYSCALE)
# i_4 = cv2.imread('imagenes/SENOS1-L.BMP',cv2.IMREAD_GRAYSCALE)

i_1 = i_1.astype(float) + 1

hist, bins = np.histogram(i_1.ravel(), bins=256, range=[0, 256])
# Grafica el histograma
plt.plot(hist, color='black')
plt.xlabel('Intensidad de Píxel')
plt.ylabel('Frecuencia')
plt.title('Histograma de la Imagen i_1')
plt.show()