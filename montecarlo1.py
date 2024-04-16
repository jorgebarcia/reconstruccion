import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

n_bi = 1000
n_po = 0
n_pb = 0

bismuto = [n_bi]
polonio = [n_po]
plomo= [n_pb]

tau_bi = 7.1
tau_po = 191

simulacion = 250  # dias
# t=np.linspace(1,simulacion,simulacion)
t=np.linspace(0,simulacion,simulacion+1) # así podemos implementar el 0


def distri_exp(tau, simulacion):
    return (1 / tau) * np.exp(-simulacion / tau)


cont_bi=0
cont_po=0
for i in range(0,simulacion):
    # hacemos un array de num aleatorios para el nucleo, size= n_bismuto
    random_bismuto = np.random.rand(bismuto[-1])

    # ahora vemos que nucleos se desintegran, si es <= se desintegra
    desin_bismuto = np.where(random_bismuto <=  distri_exp(tau_bi,i),1,0)
    # si no de desintegra --> 0 , si lo hace --> 1
    cont_bi +=desin_bismuto.sum()
    #claro por que de cada vez se me oueden desinteggrar varios

    random_polonio = np.random.rand(polonio[-1])
    desin_polonio= np.where(random_polonio <= distri_exp(tau_po,i),1,0)
    cont_po +=desin_polonio.sum()

    # num_bismuto = bismuto[-1] - desi

    nuevo_bi = bismuto[-1] - desin_bismuto.sum()
    nuevo_po = polonio[-1] - desin_polonio.sum() + desin_bismuto.sum()  # + los nuevos bismu
    nuevo_pb = plomo[-1] + desin_polonio.sum()  # + los nuevos polo

    bismuto.append(nuevo_bi)
    polonio.append(nuevo_po)
    plomo.append(nuevo_pb)

# 'ploooot'
# plt.style.use('bmh')
# plt.figure()
# plt.plot(t, bismuto, label='Bismuto')
# plt.plot(t, polonio, label='Polonio')
# plt.plot(t, plomo, label='Plomo')
# plt.xlabel('Tiempo(días)')
# plt.ylabel('# nucleos')
# plt.title('Método de montecarlo')
# plt.legend()
# plt.show()


def modelo_bismuto(t, N0, tau):
    return N0 * np.exp(-t / tau)

def modelo_polonio(t, N0, tau_bi, tau_po):
    omegax = 1/tau_bi
    omegay = 1/tau_po
    return (omegax / (omegay - omegax)) * N0 * (np.exp(-t * omegax) - np.exp(-t * omegay))

t = np.linspace(0, simulacion, simulacion + 1)

# Ajuste para el Bismuto
parametros_bi, _ = curve_fit(modelo_bismuto, t, bismuto, p0=[n_bi, tau_bi])

# Ajuste para el Polonio, suponiendo que el Polonio se forma a partir del Bismuto desintegrado
parametros_po, _ = curve_fit(modelo_polonio, t, polonio, p0=[n_bi, tau_bi, tau_po])

# Plot de los resultados
plt.figure(figsize=(10, 6))
plt.plot(t, bismuto, 'r-', label='Simulación Bismuto')
plt.plot(t, modelo_bismuto(t, *parametros_bi), 'r--', label='Ajuste Bismuto')
plt.plot(t, polonio, 'g-', label='Simulación Polonio')
plt.plot(t, modelo_polonio(t, *parametros_po), 'g--', label='Ajuste Polonio')
plt.plot(t, plomo, 'b-', label='Simulación Plomo')
plt.title('Simulación y Ajuste de la Desintegración Nuclear')
plt.xlabel('Tiempo (días)')
plt.ylabel('Número de núcleos')
plt.legend()
plt.grid(True)
plt.show()

print(f"Parámetros ajustados para Bismuto: N0 = {parametros_bi[0]:.2f}, tau = {parametros_bi[1]:.2f} días")
print(f"Parámetros ajustados para Polonio: N0 = {parametros_po[0]:.2f}, tau_bi = {parametros_po[1]:.2f}, tau_po = {parametros_po[2]:.2f}")




