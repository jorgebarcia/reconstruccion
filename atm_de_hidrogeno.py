import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('ggplot')

# plt.style.use('bmh')
plt.style.use('grayscale')
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
a0=0.529177e-10
p=2*h*h*c*c/(m*a0*a0)
print(p)

def calc_c(L,u,alpha,p=p):
    val= L * (L + 1) / (u * u) - 2/u +alpha*p
    # print(p)
    return val
def numerov(u, du, alpha,L):
    phi = np.zeros((u.size))
    psi = np.zeros_like(u)
    phi[1] = 0.00001
    psi[1] = phi[1] / (1 - du * du * calc_c(u[1], k, alpha,L) / 12)

    for i in range(2, len(u)):
        phi[i] = 2 * phi[i - 1] - phi[i - 2] + du * du * calc_c(L,u[i - 1], k, alpha) * psi[i - 1]
        psi[i] = phi[i] / (1 - du * du * calc_c(L,u[i], k, alpha) / 12)
    return phi, psi
def biseccion(alpha1, alpha2,L, tol=1e-15):
    alpha_izq = alpha1
    alpha_der = alpha2
    while True:
        if np.abs(alpha_der - alpha_izq) < tol:
            break
        _, psi_izq = numerov(u, du, alpha_izq,L)
        _, psi_der = numerov(u, du, alpha_der,L)
        print(alpha_izq,'izq', psi_izq[-1])
        print(alpha_der,'der', psi_der[-1])
        if psi_izq[-1] * psi_der[-1] < 0:

            alpha_m = (alpha_izq + alpha_der) / 2
            _, psi_m = numerov(u, du, alpha_m,L)

            if psi_izq[-1] * psi_m[-1] < 0:
                alpha_der = alpha_m
                # print('izq', alpha_izq)
                # print('m', alpha_m)
            else:
                alpha_izq = alpha_m
                # print('der', alpha_der)
                # print('m', alpha_m)
        else:
            print('No hemos encontrado un alpha')
        print('-' * 50)

    return alpha_m

def pintar_alpha_dado(alpha1, alpha2,L):
    """
    :rtype: object
    """
    alpha = biseccion(alpha1, alpha2,L, tol=1e-10)
    _, psi = numerov(u, du, alpha,L)
    plt.figure()
    plt.plot(u, psi)
    plt.title('Funcion de onda con alpha = {alpha}'.format(alpha=alpha))
    plt.show()
    return alpha

alpha = pintar_alpha_dado(0,0.1,L=0)

# alpha = pintar_alpha_dado(0.2,0.3,L=0)
# alpha = pintar_alpha_dado(0.1,0.15,L=0)
# alpha = pintar_alpha_dado(0,0.8,L=0)
# alpha = pintar_alpha_dado(alpha,0.8,L=0)

# valores de verdad
'-------------------------'
E0=-13.6
E1= -3.4
E2=-1.511
E3=-0.85

a_teo_1=E1/E0
print(a_teo_1)
a_teo_2=E2/E0
print(a_teo_2)
a_teo_3=E3/E0
print(a_teo_3)
