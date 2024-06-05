import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
from skimage.transform import resize
from skimage.util import img_as_float
from skimage.transform import resize
from mpl_toolkits.mplot3d import Axes3D

def cargar_imagen(path):
    """ Carga y normaliza una imagen """
    img = img_as_float(io.imread(path, as_gray=True))
    img = exposure.equalize_hist(img)  # Normalización del histograma
    return resize(img, (512, 512))  # Asegurar que todas las imágenes tengan el mismo tamaño

def generar_mapa_disparidad(img1, img2):
    """ Generar un mapa de disparidad simple (placeholder) """
    # Este código es un ejemplo y no realizará un cálculo real de disparidad
    return np.abs(img1 - img2)

def reconstruir_superficie(disparidades):
    """ Integrar disparidades para reconstruir la superficie 3D """
    # Sumar las disparidades ponderadas (placeholder)
    return sum(disparidades) / len(disparidades)

def visualizar_modelo_3d(superficie):
    """ Visualizar el modelo 3D de la superficie """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(superficie.shape[0]), np.arange(superficie.shape[1]))
    ax.plot_surface(X, Y, superficie, cmap='viridis')
    plt.show()

# Rutas de ejemplo a las imágenes
paths = ['calibrado/0_4-T.BMP','calibrado/0_4-B.BMP','calibrado/0_4-L.BMP',
         'calibrado/0_4-R.BMP']

# img_rutas = {'top': 'imagenes/SENOS1-T.BMP', 'bottom': 'imagenes/SENOS1-B.BMP', 'left': 'imagenes/SENOS1-L.BMP',
#              'right': 'imagenes/SENOS1-R.BMP', 'textura': 'imagenes/SENOS1-S.BMP'}

imagenes = [cargar_imagen(path) for path in paths]

# Generar mapas de disparidad
disparidades = [generar_mapa_disparidad(imagenes[i], imagenes[(i+1) % 4]) for i in range(4)]

# Reconstruir la superficie 3D
superficie_3d = reconstruir_superficie(disparidades)

# Visualizar el modelo 3D
visualizar_modelo_3d(superficie_3d)