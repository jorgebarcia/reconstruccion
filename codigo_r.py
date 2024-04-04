# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:41:02 2024

@author: rober
"""

import numpy as np
import matplotlib.pyplot as plt
# import scipy.optimize as so
from timeit import default_timer as cn

# Pozo de potencial

# Constantes
V0 = 244  # eV
a = 1e-10
me = 0.511 * 1e6 / (3 * 10 ** 8) ** 2
hbarra = 6.582e-16
k = 2 * me * (a ** 2) * V0 / (hbarra ** 2)

alpha = 0.5

# Dx = 0.001
umax = 2
u = np.linspace(0, umax, 1001)
# nmalla = np.size(u)
du = (u[0] - u[-1]) / 1001

utot = np.linspace(-umax, umax, 2002)


# Potencial C:
def C(u, alpha):
    if -0.5 < u < 0.5:
        return -k * alpha

    else:
        return k * (1 - alpha)


psipar = np.zeros(len(u))
dpsipar = np.zeros(len(u))
psipar[0] = 1
dpsipar[0] = 0

psiimpar = np.zeros(len(u))
dpsiimpar = np.zeros(len(u))
psiimpar[0] = 0
dpsiimpar[0] = 1


def funciondeondapar(alpha):
    for i in range(0, len(u) - 1):
        psipar[i + 1] = psipar[i] + dpsipar[i] * du
        dpsipar[i + 1] = dpsipar[i] + C(u[i], alpha) * psipar[i] * du

        # psiparsim=np.copy(-np.sort(psipar))
        # np.sort(psipar)

        psipartotal = np.concatenate((np.flip(psipar), psipar))
    return psipartotal


def funciondeondaimpar(alpha):
    for i in range(0, len(u) - 1):
        psiimpar[i + 1] = psiimpar[i] + dpsiimpar[i] * du
        dpsiimpar[i + 1] = dpsiimpar[i] + C(u[i], alpha) * psiimpar[i] * du

        psiimpartotal = np.concatenate((np.flip(-psiimpar), psiimpar))
    return psiimpartotal


# (np.flip(psipar)
# plt.plot(urep, funciondeonda(alpha))

alpha = np.linspace(0, 1, 10)

tol = 1e-12
# alphafinal = []

'''
def biseccion(funciondeonda,alpha,tol):
    for j in range(len(alpha)-1):
        biseccion1 = funciondeonda(alpha[j])
        biseccion2 = funciondeonda(alpha[j+1])

        if biseccion1[-1]*biseccion2[-1]>0:
            continue
        else:
            while abs(alpha[j]-alpha[j+1])>tol:
                medio = (alpha[j]+alpha[j+1])/2
                biseccionmedio = funciondeonda(medio)
                if biseccionmedio[-1]*biseccion1[-1]<0:
                    alpha[j]=medio
                else:
                    alpha[j+1]=medio
            alphafinal.append(medio)
    return alpha
'''


def biseccion(funciondeonda, a, b, tol):
    if funciondeonda(a)[-1] * funciondeonda(b)[-1] > 0:
        print('Sin raíces.')
    else:
        while (b - a) / 2 > tol:
            medio = (a + b) / 2
            if funciondeonda(medio)[-1] == 0:
                return (medio)
            elif funciondeonda(a)[-1] * funciondeonda(medio)[-1] < 0:
                b = medio
            else:
                a = medio
        return medio


# answer=biseccion(funciondeondapar, 0.5, 0.9, tol)
# print(answer)
tinicial = cn()

psi1 = funciondeondapar(biseccion(funciondeondapar, 0, 0.5, tol))
psi2 = funciondeondaimpar(biseccion(funciondeondaimpar, 0, 0.5, tol))
psi3 = funciondeondapar(biseccion(funciondeondapar, 0.5, 0.9, tol))

fig = plt.figure(figsize=(80, 80))
axes = fig.add_subplot(111)
axes.plot(utot, psi1)
axes.plot(utot, psi2)
axes.plot(utot, psi3)
axes.grid()

tfinal = cn()
print('Tiempo de ejecución: {}s'.format(np.round(tfinal - tinicial, 3)))

autovalores = [biseccion(funciondeondapar, 0, 0.5, tol), biseccion(funciondeondaimpar, 0, 0.5, tol),
               biseccion(funciondeondapar, 0.5, 0.9, tol)]
print('Los autovalores son: ' + str(autovalores))

'''
for h in range(len(alphafinal)):
    ppar = funciondeonda(alphafinal[h])
    plt.plot(urep,ppar)
'''