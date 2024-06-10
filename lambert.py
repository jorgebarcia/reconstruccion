import numpy as np

# D=create_difference_operator_forward(10)
# print(D)
# D=create_difference_operator_central(10)
# print(D)

# D=create_seven_point_difference_operator(10)
# print(D)

def create_central_difference_operator(n, order=3):
    D = np.zeros((n, n))
    if order == 3:
        for i in range(1, n - 1):
            D[i, i - 1] = -0.5
            D[i, i + 1] = 0.5
        D[0, 1] = 0.5  # Ajuste para la primera fila
        D[n - 1, n - 2] = -0.5  # Ajuste para la última fila
    # Aquí podríamos añadir otros órdenes si es necesario
    return D

# D=create_central_difference_operator(10)
# print(D)


def create_central_difference_operator_order_5(n):
    D = np.zeros((n, n))
    for i in range(2, n - 2):
        D[i, i - 2] = 1 / 12
        D[i, i - 1] = -8 / 12
        D[i, i + 1] = 8 / 12
        D[i, i + 2] = -1 / 12

    # Ajustes para las primeras dos filas y las últimas dos filas
    D[0, 0] = -25 / 12
    D[0, 1] = 4
    D[0, 2] = -3
    D[0, 3] = 4 / 3
    D[0, 4] = -1 / 4

    D[1, 0] = -1 / 4
    D[1, 1] = -25 / 12
    D[1, 2] = 4
    D[1, 3] = -3
    D[1, 4] = 4 / 3
    D[1, 5] = -1 / 4

    D[-2, -6] = 1 / 4
    D[-2, -5] = -4 / 3
    D[-2, -4] = 3
    D[-2, -3] = -4
    D[-2, -2] = 25 / 12
    D[-2, -1] = -1 / 4

    D[-1, -5] = 1 / 4
    D[-1, -4] = -4 / 3
    D[-1, -3] = 3
    D[-1, -2] = -4
    D[-1, -1] = 25 / 12

    return D
# D=create_central_difference_operator_order_5(10)
# print(D)



import numpy as np


h = 1


A = np.array([
    [1, -2*h, 1/2*4*h**2, 1/6*(-8*h**3), 1/24*(16*h**4)],
    [1, -h,1/2*h**2, 1/6*(-h**3), 1/24*(h**4)],
    [1, 0, 0, 0, 0],
    [1, h, 1/2*h**2, 1/6*(h**3), 1/24*(h**4)],
    [1, 2*h, 1/2*4*h**2, 1/6*(8*h**3), 1/24*(16*h**4)]
])
A = np.array([
    [1, -2*h, (2*h)**2 / 2, -(2*h)**3 / 6, (2*h)**4 / 24],
    [1, -h, h**2 / 2, -h**3 / 6, h**4 / 24],
    [1, 0, 0, 0, 0],
    [1, h, h**2 / 2, h**3 / 6, h**4 / 24],
    [1, 2*h, (2*h)**2 / 2, (2*h)**3 / 6, (2*h)**4 / 24]
])

b = np.array([0, 0, 1, 0, 0])

# print(A)
coeffs = np.linalg.solve(A, b)


print("Coeficientes de la matriz de diferencias centradas de orden 5:")
print(coeffs)


A = np.array([
    [1, -h, 1/2*h**2],
    [1, 0, 0],
    [1, h, 1/2*h**2]
])

b = np.array([0, 1, 0])


coeffs = np.linalg.solve(A, b)

print("Coeficientes de la matriz de diferencias centradas de orden 3:")
print(coeffs)
import math

def solve_fd_coefficients(order, h=1):
    points = order + 1
    A = np.zeros((points, points))
    b = np.zeros(points)
    b[points // 2] = 1  # Establecemos la derivada que queremos reproducir en el punto central

    for i in range(points):
        x = i - points // 2
        for j in range(points):
            A[i, j] = (x * h) ** j / math.factorial(j)
    print(A)
    coeffs = np.linalg.solve(A, b)
    return coeffs

# Ejemplo de uso
# order_3_coeffs = solve_fd_coefficients(2)  # Orden 3
# order_5_coeffs = solve_fd_coefficients(4)  # Orden 5
# print('')
# print("Coeficientes para diferencias centradas de orden 3:", order_3_coeffs)
# print("Coeficientes para diferencias centradas de orden 5:", order_5_coeffs)

print('\n --------------------- \n')

def build_diff_matrix(n, coeffs):
    order = len(coeffs)
    mid = order // 2
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(-mid, mid + 1):
            if 0 <= i + j < n:
                D[i, i + j] = coeffs[mid + j]
    return D

# Tamaño de los datos
n = 8  # Cambia esto según tus datos

# Coeficientes encontrados anteriormente
coeffs_3 = [1, 0, -2]
coeffs_5 = [1, -1.48e-16, -2.5, 2.22e-16, 6]

# Crear matrices de diferenciación
D3 = build_diff_matrix(n, coeffs_3)
D5 = build_diff_matrix(n, coeffs_5)

print("Matriz de Diferenciación de Orden 3:")
print(D3)
print("\nMatriz de Diferenciación de Orden 5:")
print(D5)


def paper(n):
    D = np.zeros((n, n))
    # Asumiendo que los coeficientes que aparecen en el paper son correctos y están distribuidos adecuadamente
    center_coeffs = [1, -8, 0, 8, -1]  # Coeficientes para puntos centrales

    if n < 5:
        raise ValueError("n debe ser al menos 5 para este esquema")

    # Aplicar coeficientes centrales para la mayoría de las filas
    for i in range(2, n - 2):
        D[i, i - 2:i + 3] = center_coeffs

    # Ajustes para los primeros y últimos puntos, asumiendo simetría y usando un esquema simplificado
    D[0, :5] = [1, -4, 6, -4, 1]  # Puedes ajustar esto según el paper si es necesario
    D[1, :5] = [-1, 4, -5, 4, -1]  # Ajuste para el segundo punto
    D[-2, -5:] = [-1, 4, -5, 4, -1]  # Ajuste simétrico para el penúltimo punto
    D[-1, -5:] = [1, -4, 6, -4, 1]  # Ajuste simétrico para el último punto

    return D


# Tamaño de la matriz
n = 5  # Ejemplo para una matriz grande
# D = paper(n)
# print(D)


D= 0.5 * np.array([
    [-3, 4, -1, 0, 0],
    [-1, 0, 1, 0, 0],
    [0, -1, 0, 1, 0],
    [0, 0, -1, 0, 1],
    [0, 0, 1, -4, 3]
])
print(D)
A_T=D.T
is_orthonormal = np.allclose(np.dot(A_T, D), np.identity(D.shape[0]))

B=D@A_T
print(B)

print("¿Es A ortonormal?", is_orthonormal)


