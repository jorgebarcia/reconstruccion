import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

L=np.linspace(0,0.1,1000)
sigma=59.6e6
A=0.00001

G=sigma*A/np.clip(L,1e-4,np.inf)

plt.plot(L,G)
plt.xlabel("L (m)")
plt.ylabel("G (m/Ωmm²)")
plt.show()