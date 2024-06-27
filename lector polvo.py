'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('ggplot')

# Cargar datos
# df = pd.read_csv('imagenes/polvo8t.csv')
df = pd.read_csv('imagenes/Results 16.csv')
# df = pd.read_csv('imagenes/feret.csv')

# Ajustar los datos a una distribución normal
mu, std = norm.fit(df['Length'])
# mu, std = norm.fit(df['Feret'])

# Configurar los colores según tu póster
color_histograma = '#214579'  # Un azul oscuro
color_fit = '#d7191c'  # Un rojo brillante

# Plotear el histograma de los datos
plt.figure(figsize=(10, 6))
plt.hist(df['Length'], bins=30, density=False, alpha=0.7, color=color_histograma)

# Plotear la distribución gaussiana sobre el histograma de frecuencias
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std) * len(df['Length']) * (xmax - xmin) / 30
plt.plot(x, p, color=color_fit, linewidth=2)

# Agregar líneas de puntos para la media y desviaciones estándar
plt.axvline(mu, color='k', linestyle='dashed', linewidth=1)
plt.axvline(mu + std, color='k', linestyle='dotted', linewidth=1)
plt.axvline(mu - std, color='k', linestyle='dotted', linewidth=1)

# Agregando texto para la media y desviaciones con sus valores
plt.text(mu, plt.ylim()[1]*0.9, r'$\mu = {:.2f}$ $(\mu m)$'.format(mu), horizontalalignment='center',fontsize=15)
plt.text(mu + std, plt.ylim()[1]*0.85, r'$\mu + \sigma = {:.2f}$ $(\mu m)$'.format(mu + std), horizontalalignment='center',fontsize=15, color=color_fit)
plt.text(mu - std, plt.ylim()[1]*0.85, r'$\mu - \sigma = {:.2f}$ $(\mu m)$'.format(mu - std), horizontalalignment='center', fontsize=15 ,color=color_fit)
plt.tick_params(axis='both', which='major', labelsize=13)


# Titulo y etiquetas
plt.title('Radio partículas Acero 316L',fontsize=18,fontweight='bold')
plt.xlabel(r'Radio $(\mu m)$',fontsize=16,fontweight='bold')
plt.ylabel('Frecuencia',fontsize=16,fontweight='bold')
plt.grid(False)
# Mostrar gráfico
plt.show()


#F8B285'''

'''
import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('imagenes/POLVO-1.BMP', cv2.IMREAD_GRAYSCALE)

# Suavizar la imagen para reducir el ruido
imagen_suavizada = cv2.GaussianBlur(imagen, (9, 9), 10)

# Detección de círculos mediante la transformada de Hough
circulos = cv2.HoughCircles(imagen_suavizada, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=30)

# Verificar si se encontraron círculos
if circulos is not None:
    circulos = np.uint16(np.around(circulos))
    print(f"Se encontraron {len(circulos[0, :])} círculos")

    # Dibujar los círculos detectados en la imagen original
    for i in circulos[0, :]:
        cv2.circle(imagen, (i[0], i[1]), i[2], (255, 0, 0), 2)
        cv2.circle(imagen, (i[0], i[1]), 2, (255, 0, 255), 3)

    # Mostrar la imagen con los círculos
    cv2.imshow('Círculos Detectados', imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se encontraron círculos")

'''

import cv2
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
from scipy.stats import norm
import pandas as pd  # Importar panda

# Cargar la imagen
imagen = cv2.imread('imagenes/POLVO-1.BMP', cv2.IMREAD_GRAYSCALE)

# Suavizar la imagen para reducir el ruido
# imagen_suavizada = cv2.GaussianBlur(imagen, (9, 9), 2)
imagen_suavizada = cv2.medianBlur(imagen, 5)

# _, imagen_suavizada = cv2.threshold(imagen_suavizada, 80, 255, cv2.THRESH_BINARY)
# _, imagen_suavizada = cv2.threshold(imagen_suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Alternativa: Umbralización adaptativa
# imagen_suavizada = cv2.adaptiveThreshold(imagen_suavizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                            cv2.THRESH_BINARY, 11, 6)


# Detección de círculos mediante la transformada de Hough
circulos = cv2.HoughCircles(imagen_suavizada, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=70, param2=25, minRadius=10, maxRadius=45)

if circulos is not None:
    circulos = np.uint16(np.around(circulos))

    # Dibujar los círculos detectados en la imagen original
    imagen_con_circulos = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    for i in circulos[0, :]:
        cv2.circle(imagen_con_circulos, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv2.circle(imagen_con_circulos, (i[0], i[1]), 2, (0, 255, 0), 3)

    # Mostrar la imagen con los círculos
    plt.figure(figsize=(5, 4))
    plt.imshow(imagen_con_circulos)
    plt.title('Partículas detectadas')
    plt.axis('off')
    plt.show()

    # Calcular los diámetros en micrómetros
    diametros_pixeles = [2 * c[2] for c in circulos[0, :]]
    factor_conversion = 5.0362  # Píxeles por micrómetro
    diametros_micrometros = [d / factor_conversion for d in diametros_pixeles]

    # Crear DataFrame y exportar a CSV
    df = pd.DataFrame(diametros_micrometros, columns=['Diametro_Micrometros'])
    df.to_csv('diametros_particulas.csv', index=False)

    # Histograma con ajuste gaussiano y marcas de D10, D50, y D90
    plt.figure(figsize=(10, 6))
    (mu, sigma) = norm.fit(diametros_micrometros)
    n, bins, patches = plt.hist(diametros_micrometros, bins=30, density=True, alpha=0.7, color='blue')
    y = norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=2)
    D10 = np.percentile(diametros_micrometros, 10)
    D50 = np.percentile(diametros_micrometros, 50)
    D90 = np.percentile(diametros_micrometros, 90)
    plt.axvline(D10, color='green', linestyle='dashed', linewidth=1.5)
    plt.axvline(D50, color='purple', linestyle='dashed', linewidth=1.5)
    plt.axvline(D90, color='green', linestyle='dashed', linewidth=1.5)
    plt.text(D10, plt.ylim()[1] * 0.95, f'D10: {D10:.2f} μm', horizontalalignment='center', color='green')
    plt.text(D50, plt.ylim()[1] * 0.90, f'D50: {D50:.2f} μm', horizontalalignment='center', color='purple')
    plt.text(D90, plt.ylim()[1] * 0.95, f'D90: {D90:.2f} μm', horizontalalignment='center', color='green')
    plt.title('Histograma de Distribución de Diámetros de Partículas')
    plt.xlabel('Diámetro (micrómetros)')
    plt.ylabel('Densidad de probabilidad')
    plt.grid(True)
    plt.show()

else:
    print("No se encontraron círculos")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm

# Configuración inicial
np.random.seed(0)

# Generar datos de múltiples distribuciones con sesgo
data1 = skewnorm.rvs(a=5, loc=7, scale=1, size=300)  # Distribución sesgada hacia la derecha
data2 = skewnorm.rvs(a=-5, loc=10, scale=1.5, size=700)  # Distribución sesgada hacia la izquierda
data = np.concatenate([data1, data2])

# Agregar ruido para más variabilidad
data += np.random.normal(0, 0.3, size=data.size)

# Histograma de los datos
plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(data, bins=30, density=True, alpha=0.65, color='blue', label='Distribución Simulada')

# Ajustar y trazar una curva gaussiana
param = norm.fit(data)
x = np.linspace(min(data), max(data), 100)
pdf_fitted = norm.pdf(x, *param)
plt.plot(x, pdf_fitted, 'r-', label='Ajuste')

# Calcular y marcar D10, D50 y D90
D10 = np.percentile(data, 10)
D50 = np.percentile(data, 50)
D90 = np.percentile(data, 90)

plt.axvline(D10, color='green', linestyle='dashed', linewidth=1.5)
plt.axvline(D50, color='green', linestyle='dashed', linewidth=1.5)
plt.axvline(D90, color='green', linestyle='dashed', linewidth=1.5)

plt.text(D10, plt.ylim()[1] * 0.95, f'D10: {D10:.2f} μm', horizontalalignment='center', color='green')
plt.text(D50, plt.ylim()[1] * 0.90, f'D50: {D50:.2f} μm', horizontalalignment='center', color='purple')
plt.text(D90, plt.ylim()[1] * 0.95, f'D90: {D90:.2f} μm', horizontalalignment='center', color='red')

plt.title('Distribución Simulada Mejorada con Asimetría')
plt.xlabel('Diámetro (μm)')
plt.ylabel('Densidad de probabilidad')
plt.legend()
plt.grid(False)
plt.show()

