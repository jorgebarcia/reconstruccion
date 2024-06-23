from estoyhastaloscojones import Cargarimagenes,Ecualizacion, Reconstruccion
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

# img_rutas = {'top': 'imagenes/4-C-B.BMP', 'bottom': 'imagenes/4-C-T.BMP', 'left': 'imagenes/4-C-L.BMP',
#              'right': 'imagenes/4-C-R.BMP', 'textura': 'imagenes/4-C-S.BMP'}


# img_rutas = {'top': 'imagenes/6-C-B.BMP', 'bottom': 'imagenes/6-C-T.BMP', 'left': 'imagenes/6-C-R.BMP',
#              'right': 'imagenes/6-C-L.BMP', 'textura': 'imagenes/6-C-S.BMP'}
img_rutas = {'top': 'imagenes/6M-C-B.BMP', 'bottom': 'imagenes/6M-C-T.BMP', 'left': 'imagenes/6M-C-L.BMP',
              'right': 'imagenes/6M-C-R.BMP', 'textura': 'imagenes/6M-C-S.BMP'}




cargar = Cargarimagenes(img_rutas)
cargar.upload_img(img_rutas)
ecu=Ecualizacion(cargar)
reconstruir = Reconstruccion(cargar)
sdx,sdy = reconstruir.calculo_gradientes(1,1,eps=1e-5, ver=False)

m=960
n=1280

Lx, _ = reconstruir.diffmat2(n - 1, (0, n - 1))
Ly, _ = reconstruir.diffmat2(m - 1, (0, m - 1))

Sx = reconstruir.segundadif(n)
Sy = reconstruir.segundadif(m)
# Z_t = self.reg_tikhonov(Lx, Ly, Sx, Sy, lamb=1e-2)


def calculate_residuals_and_solution_norm(Z, error):
    residual_norm = np.linalg.norm(error)
    solution_norm = np.linalg.norm(Z)
    # print('tiene q ser lo mismo q error tiki:', A @ Z + Z @ B + C)
    print('residual norm: ', residual_norm)
    print('solution norm: ', solution_norm)
    return residual_norm, solution_norm

# lambdas = np.logspace(-6, 2, 7)  # Valores de lambda desde 10^-4 hasta 10^1
# lambdas=[1e-3,1e-2,1e-1,2e-1,3e-1]
lambdas = np.logspace(-4, 0, 30)
print(lambdas)

# lambdas=[1e-6 ,1e-5 ,1e-4, 1e-3, 1e-2, 1e-1,2e-1, 1e0]

residuals = []
solutions = []
print('lambdas: ', lambdas)
contador=0
for lamb in lambdas:
    print('iteracion:',contador)
    Z,error = reconstruir.reg_tikhonov(sdx,sdy,Lx, Ly, Sx, Sy, lamb)
    res_norm, sol_norm = calculate_residuals_and_solution_norm(Z,error)
    residuals.append(res_norm)
    solutions.append(sol_norm)
    contador +=1


plt.figure()
plt.loglog(residuals, solutions, marker='o')
for i in range(len(lambdas)):
    plt.loglog(residuals[i],solutions[i],'o', label=f'λ={lambdas[i]}')

plt.xlabel('Norma de los Residuos')
plt.ylabel('Norma de la Solución')
plt.grid()
# plt.legend(loc='best')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

residuals = np.array(residuals)
solutions = np.array(solutions)