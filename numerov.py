import numpy as np
import matplotlib.pyplot as plt

'''
['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh',
 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8',
 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette',
 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper',
 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white',
 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
'''
# plt.style.use('seaborn-v0_8')
plt.style.use('seaborn-v0_8-poster')

a = 1e-10
m = 0.511e6
V0 = 244
h = 6.582e-16
c = 3e8
k = (2 * m * a * a * V0) / (h * h * c * c)

npuntos = 1000

x = np.linspace(-2e-10, 2e-10, npuntos)
u = x / a
du = np.abs(u[0] - u[1])


# phi = np.zeros((u.size))
# psi = np.zeros_like(u)

def calc_c(u, k, alpha):
    if -0.5 < u < 0.5:
        c = -k * alpha
    else:
        c = k * (1 - alpha)
    return c

def normalize(funcion):
    # norma=0
    # for i in range(1,len(funcion)):
    #     norma=norma +funcion[i]*funcion[i]
    # norma=np.sqrt(norma)
    # print(norma)
    max=np.max(funcion)

    # fun_norm=funcion/norma
    return max

def numerov(u, du, alpha):
    phi = np.zeros((u.size))
    psi = np.zeros_like(u)
    phi[1] = 0.00001
    psi[1] = phi[1] / (1 - du * du * calc_c(u[1], k, alpha) / 12)

    for i in range(2, len(u)):
        phi[i] = 2 * phi[i - 1] - phi[i - 2] + du * du * calc_c(u[i - 1], k, alpha) * psi[i - 1]
        psi[i] = phi[i] / (1 - du * du * calc_c(u[i], k, alpha) / 12)
        # print(u[i], phi[i])

    return phi, psi

def biseccion(alpha1, alpha2, tol=1e-15):
    alpha_izq = alpha1
    alpha_der = alpha2
    while True:
        if np.abs(alpha_der - alpha_izq) < tol:
            break
        _, psi_izq = numerov(u, du, alpha_izq)
        _, psi_der = numerov(u, du, alpha_der)
        print(alpha_izq)
        print(alpha_der)
        # print('')
        print('izq', psi_izq[-1])
        print('der', psi_der[-1])
        if psi_izq[-1] * psi_der[-1] < 0:

            alpha_m = (alpha_izq + alpha_der) / 2
            _, psi_m = numerov(u, du, alpha_m)

            if psi_izq[-1] * psi_m[-1] < 0:
                alpha_der = alpha_m
                print('izq', alpha_izq)
                print('m', alpha_m)
                # print(alpha_izq)
                # print('wf_1', wf_1[-1])

            else:
                alpha_izq = alpha_m
                print('der', alpha_der)
                print('m', alpha_m)
                # print('wf_2',wf_2[-1])
                # print(alpha_der)
        else:
            print('No hemos encontrado un alpha')
        print('-' * 50)

    return alpha_m


def pintar_alpha_dado(alpha1, alpha2):
    """

    :rtype: object
    """
    alpha = biseccion(alpha1, alpha2, tol=1e-10)
    _, psi = numerov(u, du, alpha)
    plt.figure()
    plt.plot(u, psi/normalize(psi))
    plt.title('Funcion de onda con alpha = {alpha}'.format(alpha=alpha))
    plt.show()
    return alpha

alpha = pintar_alpha_dado(0, 0.2)
alpha2 = pintar_alpha_dado(0.2, 0.4)
alpha3 = pintar_alpha_dado(0.4, 1)

def pinto_todo():
    alpha=biseccion(0,0.2,tol=1e-10)
    _, psi1 = numerov(u, du, alpha)
    plt.plot(u,psi1/normalize(psi1),label=r'$\alpha$={}'.format(alpha))

    alpha2 = biseccion(0.2,0.4,tol=1e-10)
    _, psi2 = numerov(u, du, alpha2)
    plt.plot(u,psi2/normalize(psi2),label=r'$\alpha$={}'.format(alpha2))

    alpha3 = biseccion(0.4,1)
    _, psi3 = numerov(u, du,alpha3)
    plt.plot(u,psi3/normalize(psi3),label=r'$\alpha$={}'.format(alpha3))
    plt.legend()
    plt.show()

pinto_todo()
