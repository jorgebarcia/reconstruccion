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
dx = np.abs(x[0] - x[1]) / len(x)

wf = np.zeros((u.size))
wf[0] = 1
dwf = np.zeros_like(wf)
dwf[0] = 0


def calc_c(u, k, alpha):
    if -0.5 < u < 0.5:
        c = -k * alpha
    else:
        c = k * (1 - alpha)
    return c

def wave_calc_par(wf, dx, u, alpha):
    dwf = np.zeros_like(wf)
    wf[0] = 1
    dwf[0] = 0

    for i in range(len(u) - 1):
        c = calc_c(u[i], k, alpha)
        wf[i + 1] = wf[i] + dwf[i] * (u[i + 1] - u[i])
        dwf[i + 1] = dwf[i] + c * wf[i] * (u[i + 1] - u[i])

    u_neg = -u[1:][::-1]
    wf_neg = wf[1:][::-1]
    dwf_neg = dwf[1:][::-1]

    u_t = np.concatenate((u_neg, u))
    wf_t = np.concatenate((wf_neg, wf))
    dwf_t = np.concatenate((dwf_neg, dwf))

    return wf_t, dwf_t, u_t

def wave_calc_impar(wf, dx, u, alpha):
    dwf = np.zeros_like(wf)
    wf[0] = 0
    dwf[0] = 1

    for i in range(len(u) - 1):
        c = calc_c(u[i], k, alpha)
        wf[i + 1] = wf[i] + dwf[i] * (u[i + 1] - u[i])
        dwf[i + 1] = dwf[i] + c * wf[i] * (u[i + 1] - u[i])

    u_neg = -u[1:][::-1]
    wf_neg = -wf[1:][::-1]
    dwf_neg = dwf[1:][::-1]

    u_t = np.concatenate((u_neg, u))
    wf_t = np.concatenate((wf_neg, wf))
    dwf_t = np.concatenate((dwf_neg, dwf))

    return wf_t, dwf_t, u_t
# tol = 1e-5

def biseccion(alpha1,alpha2,par=True,tol=1e-15):
    alpha_izq = alpha1
    alpha_der = alpha2
    while True:
        if np.abs(alpha_der-alpha_izq) < tol:
            break
        if par:
            wf_1, _, _ = wave_calc_par(wf, dx, u, alpha_izq)
            wf_2, _, _ = wave_calc_par(wf, dx, u, alpha_der)
        else:
            wf_1, _, _ = wave_calc_impar(wf, dx, u, alpha_izq)
            wf_2, _, _ = wave_calc_impar(wf, dx, u, alpha_der)
        print(alpha_izq)
        print(alpha_der)
        print('')

        if wf_1[-1] * wf_2[-1] < 0:

            alpha_m = (alpha_izq + alpha_der) / 2
            if par:
                wf_m, _, _ = wave_calc_par(wf, dx, u, alpha_m)
            else:
                wf_m, _ , _= wave_calc_impar(wf, dx, u, alpha_m)
            print('wf_1', wf_1[-1])
            print('wf_2', wf_2[-1])
            print('wf_m', wf_m[-1])
            print('')

            if wf_1[-1] * wf_m[-1] < 0:
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

# print(alpha_bueno)
#
# wf_t,dwf_t,u_t=wave_calc(wf,dx,u,alpha_bueno)
#
# plt.figure()
# plt.plot(u_t,wf_t)
# plt.show()

def pintar_alpha_dado(alpha1,alpha2,paridad):
    alpha = biseccion(alpha1,alpha2,par = paridad,tol = 1e-10)

    if paridad:
        wf_t, dwf_t, u_t = wave_calc_par(wf, dx, u, alpha)
    else:
        wf_t, dwf_t, u_t = wave_calc_impar( wf, dx, u, alpha)
    plt.figure()
    plt.plot(u_t, wf_t)
    plt.title('Funcion de onda con alpha = {alpha}'.format(alpha = alpha))
    plt.show()
    return alpha

alpha1=0
alpha2=np.pi*np.pi/k
# alpha = np.linspace(0, 1, 10)
# alpha=pintar_alpha_dado(alpha1,alpha2,True)

alpha = 0.1
# print(alpha)
# print(alpha2*3) #0.45

'Autovalor impar'
alpha=pintar_alpha_dado(0.1,0.40,False)

'segundo autovalor'
# alpha =  0.4
# alpha=pintar_alpha_dado(alpha,alpha2*6,True)





