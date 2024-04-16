'''
Métodos Numéricos y sus Aplicaciones a la Física
Física Nuclear

Pozo de Potencial
Método de Numerov

Juan Lorenzo Campos
71191645K
UO285898
'''

# Importamos los paquetes necesarios
import numpy as np
import matplotlib.pyplot as plt
import time

# Constantes y parámetros del sistema
c = 3e8  # m/s - Velocidad luz
a = 1e-10  # m - radio de Bohr
m = 0.511e6 / (c ** 2)  # eV·s^2/m^2 - Masa electrón
V0 = 244  # eV - Potencial inicial
hb = 6.582e-16  # eV·s - h barra

# Hallamos k y mallado de alpha
k = (2 * m * a ** 2 * V0) / (hb ** 2)
minalph = 0
maxalph = 1
ptsalph = 100
alphamallado = np.linspace(minalph, maxalph, ptsalph)

# Tolerancia
tolerancia = 1e-50

# Mallado
uMin = -2
uMax = 2
nPasos = 2002
delta_u = (uMax - uMin) / nPasos
u = np.linspace(uMin, uMax, nPasos)

# Empiezo a contar el tiempo
t0 = time.time()


# Definimos las funciones

# Calculo funcion fr
def fr(L, r, alfa):
    f = L * (L + 1) / (r ** 2) + 2((a * r) - alfa)
    return f


# Calculo funcion C
def C(u, alpha):
    if -0.5 < u < 0.5:
        return -k * alpha
    else:
        return k * (1 - alpha)


# Calculo funcion C (misma que alfa)
def f(u, alpha):
    if -0.5 < u < 0.5:
        return -k * alpha
    else:
        return k * (1 - alpha)


def funcion_de_onda(alpha, n=nPasos):
    psi = np.zeros(n)
    phi = np.zeros(n)

    # condiciones de contorno
    psi[0] = 0
    psi[1] = 0.1

    phi[0] = 0
    phi[1] = psi[1] * (1 - delta_u ** 2 * f(u[1], alpha) / 12)

    for i in range(1, nPasos - 1):
        phi[i + 1] = 2 * phi[i] - phi[i - 1] + delta_u ** 2 * C(u[i], alpha) * psi[i]
        psi[i + 1] = phi[i + 1] / (1 - delta_u ** 2 * C(u[i + 1], alpha) / 12)

    return psi


# Resolucion numerica de las ecuaciones
def Biseccion(ALFA, tolerancia=1e-15):
    alfa = np.array([])
    for j in range(len(ALFA) - 1):
        psi1 = funcion_de_onda(ALFA[j])[-1]
        psi2 = funcion_de_onda(ALFA[j + 1])[-1]

        if psi1 * psi2 <= 0:
            # print('cambio de signo')
            alfa1 = ALFA[j]
            alfa2 = ALFA[j + 1]
            while abs(alfa1 - alfa2) > tolerancia:
                alfa_medio = (alfa1 + alfa2) / 2
                psi2 = funcion_de_onda(alfa_medio)[-1]
                if psi1 * psi2 <= 0:
                    alfa2 = alfa_medio
                else:
                    alfa1 = alfa_medio
            alfa = np.append(alfa, alfa1)
    return alfa


# Hallamos alfa
alfa = Biseccion(alphamallado)

# Cerramos tiempo
tf = time.time()

fig1 = plt.figure()
fig1.suptitle('Ecuación de Schrödinger. Pozo de potencial. Numerov')
ax1 = fig1.add_subplot()
ax1.set_xlabel('u = x/a')
ax1.set_ylabel('$\Psi_n$ (u)')
for ind in range(0, len(alfa)):
    ax1.plot(u, funcion_de_onda(alfa[ind]) / max(abs(funcion_de_onda(alfa[ind]))), label='n = ' + str(ind + 1))

plt.legend()
plt.show()

print(u'Ha estando ejecutándose durante (s):', tf - t0)
print('Los autovalores son:', alfa)