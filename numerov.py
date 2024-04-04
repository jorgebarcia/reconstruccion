import numpy as np
import matplotlib.pyplot as plt

a = 1e-10
m = 0.511e6
V0 = 244
h = 6.582e-16
c = 3e8
k = (2 * m * a * a * V0) / (h * h * c * c)

npuntos = 1000

x = np.linspace(0, 2e-10, npuntos)
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

def numerov(u, du, alpha):
    phi = np.zeros((u.size))
    psi = np.zeros_like(u)
    phi[1]=0.00001
    psi[1]=phi[1]/(1-du*du*calc_c(u[1],k,alpha)/12)

    for i in range(2,len(u)-1):
        phi[i]=2*phi[i]-phi[i-1]+du*du*calc_c(u[i],k,alpha)*psi[i-1]
        psi[i] = phi[i] / (1 - du * du * calc_c(u[i], k, alpha) / 12)

    return phi, psi

# alpha=np.linspace(0, 1, 10)
alpha=1
phi,psi=numerov(u, du,alpha)

def biseccion(alpha1,alpha2,tol=1e-15):
    alpha_izq = alpha1
    alpha_der = alpha2
    while True:
        if np.abs(alpha_der-alpha_izq) < tol:
            break
        _, psi_izq = numerov(u,du,alpha_izq)
        _, psi_der = numerov(u,du,alpha_der)
        print(alpha_izq)
        print(alpha_der)
        # print('')

        if psi_izq[-1] * psi_der[-1] < 0:

            alpha_m = (alpha_izq + alpha_der) / 2
            _, psi_m = numerov(u,du,alpha_m)
            # print('wf_1', wf_1[-1])
            # print('wf_2', wf_2[-1])
            # print('wf_m', wf_m[-1])
            # print('')

            if psi_izq[-1] * psi_m[-1] < 0:
                alpha_der = alpha_m
                print('izq', alpha_izq)
                print('m',alpha_m)
                # print(alpha_izq)
                # print('wf_1', wf_1[-1])

            else:
                alpha_izq = alpha_m
                print('der',alpha_der)
                print('m', alpha_m)
                # print('wf_2',wf_2[-1])
                # print(alpha_der)
        else:
            print('No hemos encontrado un alpha')
        print('-' * 50)

    return alpha_m



def pintar_alpha_dado(alpha1,alpha2):
    alpha = biseccion(alpha1,alpha2,tol = 1e-10)
    _,psi=numerov(u,du,alpha)
    plt.figure()
    plt.plot(u, psi)
    plt.title('Funcion de onda con alpha = {alpha}'.format(alpha = alpha))
    plt.show()
    return alpha

alpha=pintar_alpha_dado(0,0.35)