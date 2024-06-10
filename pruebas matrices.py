import numpy as np
from matplotlib import pyplot as plt


def ope_dif(n, h):

    D = np.zeros((n, n))


    for i in range(0, n):
        D[i, i] = -1
        D[i - 1, i] = 1


    D[-1,- 1] = 1
    D[-1,0]=0
    D[-1,-2]=-1

    D = D / h
    return D








def ope_dif2(n, h):
    D = np.zeros((n, n))
    for i in range(0, n):
        D[i - 1, i] = 1
        D[i,i-1]= -1

    D[-1, 0] = 0
    D[0,-1]=0

    D[0,1]=4
    D[-1,-2]=-4

    D[0,2]=-1
    D[-1,-3]=1

    D[0,0]=-3
    D[-1,-1]=3


    D = D/2
    return D

n =5
h = 1

# Lx = ope_dif(n, h)
Lx = ope_dif2(n, h)

# print(Dx)
# A=Dx@Dx.T

I=np.eye(n)
print('\n Lx:')
print(Lx)

# print('\n I:')
# print(I)

# A=Lx@I
# print('\n producto:')
# print(A)

ort=Lx.T@Lx

print('\n Ort:')
print(ort)

def create_v(n):
    v=np.ones((n,1))
    v[0]=1+np.sqrt(n)
    return v

def create_I(n):
    I = np.eye(n)
    return I

def calc_P(n):
    v = np.ones((n, 1))
    v[0] = 1 + np.sqrt(n)
    I = np.eye(n)

    vvt= v @ v.T
    print('')
    print(vvt)
    vtv = v.T @ v
    print(vtv)
    P = I - 2 * (vvt/vtv)
    return P


# v=create_v(n)
# print(v)

# I=create_I(n)

#luego tenemos p que es: P=I-2*()/()
# def calc_P(n):
#     vvt= v @ v.T
#     print('')
#     print(vvt)
#     vtv = v.T @ v
#     print(vtv)
#     P = I - 2 * (vvt/vtv)
#     return P

# v=create_v(n)
P= calc_P(n)

# print('\n v:')
# print(v)

print('\n P')
print(P)

Lxg = Lx@P

tol=1e-5

Lxg[np.abs(Lxg) < tol] = 0
print('\n Lx gorro:')
print(Lxg)

que= P @ P.T
que[np.abs(que) < tol] = 0
print('\nque')
print(que)



def orto(A, tol=1e-8):
    n, m = A.shape
    if n != m:
        return False  # La matriz debe ser cuadrada para ser ortogonal
    I = np.eye(n)
    if np.allclose(A.T @ A, I, atol=tol):
        return True
    else:
        return False
def cond(A):
    U, S, Vt = np.linalg.svd(A)
    if S.min() == 0:
        return (np.inf, S)  # Retorna una tupla con infinito y los valores singulares
    else:
        condition = S.max() / S.min()
        return (condition, S)  # Consistentemente retorna una tupla

def plot_cond(A):
    # Calcular los valores singulares sin calcular los vectores U y V
    singular_values = np.linalg.svd(A, compute_uv=False)

    # Graficar los valores singulares en escala logarítmica
    plt.figure(figsize=(12, 7))
    plt.plot(singular_values, marker='o', linestyle='-')
    plt.yscale('log')  # Establece la escala del eje Y a logarítmica
    plt.title('Valores Singulares de la Matriz en Escala Logarítmica')
    plt.xlabel('Índice')
    plt.ylabel('Valor Singular (escala logarítmica)')
    plt.grid(True)
    plt.show()

# Ejemplo de uso
print("Is the matrix orthogonal?", orto(Lx))
c,_=cond(Lx)
print("Condition number of the matrix:", c)

print("Is the matrix orthogonal?", orto(Lxg))
d,_=cond(Lxg)
print("Condition number of the matrix:", d)

print("Is the matrix orthogonal?", orto(que))
e,_=cond(que)
print("Condition number of the matrix:", e)


trans=P.T@Lx.T
trans[np.abs(trans) < tol] = 0

print(Lxg.T)
print(trans)

print(Lxg.T@Lxg)
print(trans@Lxg)



