# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Asegúrate de cambiar 'ruta/a/tu/archivo.csv' por la ruta real donde guardaste tu archivo CSV
# df = pd.read_csv('imagenes/polvo8t.csv')
#
# # Ajustamos los bins del histograma al rango máximo real observado
# plt.figure(figsize=(10, 6))
# plt.hist(df['Length'], bins=range(int(df['Length'].min()), int(df['Length'].max()) + 1), color='green', alpha=0.7)
# plt.title('Histograma de Longitud de Partículas')
# plt.xlabel('Longitud')
# plt.ylabel('Frecuencia')
# plt.grid(True)
# plt.show()
#
#
# media = df['Length'].mean()
# desviacion_std = df['Length'].std()
# mediana = df['Length'].median()
# minimo = df['Length'].min()
# maximo = df['Length'].max()
# q1 = df['Length'].quantile(0.25)
# q3 = df['Length'].quantile(0.75)
# iqr = q3 - q1
# coef_variacion = desviacion_std / media if media != 0 else 0
#
# # Mostrar los resultados
# print(f"Media: {media}")
# print(f"Desviación estándar: {desviacion_std}")
# print(f"Mediana: {mediana}")
# print(f"Mínimo: {minimo}")
# print(f"Máximo: {maximo}")
# print(f"Primer cuartil (Q1): {q1}")
# print(f"Tercer cuartil (Q3): {q3}")
# print(f"Rango intercuartílico (IQR): {iqr}")
# print(f"Coeficiente de variación: {coef_variacion:.2f}")

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


#F8B285
