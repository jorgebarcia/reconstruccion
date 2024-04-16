import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from mpl_toolkits.mplot3d import Axes3D  # Importación para gráficos 3D

# Función para leer y convertir imagen a arreglo numpy
import tkinter as tk
from tkinter import filedialog

# Inicializar Tkinter y ocultar la ventana principal
root = tk.Tk()
root.withdraw()

# Función para abrir el diálogo de selección de archivo y leer imagen
def seleccionar_y_leer_imagen():
    archivo = filedialog.askopenfilename(title='Seleccionar archivo de imagen', filetypes=[('Bitmap files', '*.bmp')])
    with Image.open(archivo) as img:
        return np.asarray(img, dtype=np.float64) + 1

# Pedir al usuario cargar las imágenes mediante un diálogo de selección de archivos
I1 = seleccionar_y_leer_imagen()
I2 = seleccionar_y_leer_imagen()
I3 = seleccionar_y_leer_imagen()
I4 = seleccionar_y_leer_imagen()

# Calcular matrices z
Tamano = I1.shape
z1, z2, z3, z4 = np.zeros(Tamano), np.zeros(Tamano), np.zeros(Tamano), np.zeros(Tamano)

for i in range(Tamano[0]):
    for j in range(1, Tamano[1]):
        z1[i, j] = (I1[i, j] - I2[i, j]) / (I1[i, j] + I2[i, j]) + z1[i, j-1]

for i in range(Tamano[0]):
    for j in range(Tamano[1]-2, -1, -1):
        z3[i, j] = (-I1[i, j] + I2[i, j]) / (I1[i, j] + I2[i, j]) + z3[i, j+1]

for j in range(Tamano[1]):
    for i in range(1, Tamano[0]):
        z2[i, j] = (I3[i, j] - I4[i, j]) / (I3[i, j] + I4[i, j]) + z2[i-1, j]

for j in range(Tamano[1]):
    for i in range(Tamano[0]-2, -1, -1):
        z4[i, j] = (-I3[i, j] + I4[i, j]) / (I3[i, j] + I4[i, j]) + z4[i+1, j]

z = -z1 - z2 - z3 - z4

# Calibración basada en aumentos
Aumentos = float(input('Introduce los aumentos de la imagen: '))
Coef50, Coef150, Coef300 = 0.0120, 0.0024, 0.0021
if 50 < Aumentos < 150:
    CalibracionZ = ((Aumentos-50)*(Coef150-Coef50)/(150-50))+Coef50
elif Aumentos > 150:
    CalibracionZ = ((Aumentos-150)*(Coef300-Coef150)/(300-150))+Coef150

# Coordenadas para la visualización
CalibracionX = np.linspace(0, 0.5, Tamano[1]) / 126
CalibracionY = np.linspace(0, 0.5, Tamano[0]) / 126
X, Y = np.meshgrid(CalibracionX, CalibracionY)

# Suavizado y visualización
h = np.ones((20, 20)) / 400
suavizado = uniform_filter(2 * z * CalibracionZ, size=20)

# Crear figura para visualización 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, suavizado, cmap='viridis')

ax.set_xlabel('Calibracion X')
ax.set_ylabel('Calibracion Y')
ax.set_zlabel('Suavizado')
plt.show()
