import numpy as np

def energy_function(z_flat, s_dx, s_dy, lambda_reg):
    z = z_flat.reshape(s_dx.shape)
    dz_dx = np.gradient(z, axis=1)
    dz_dy = np.gradient(z, axis=0)
    data_fidelity = np.sum((dz_dx - s_dx) ** 2 + (dz_dy - s_dy) ** 2)
    laplacian = np.gradient(np.gradient(z, axis=1), axis=1) + np.gradient(np.gradient(z, axis=0), axis=0)
    regularization = np.sum(laplacian ** 2)
    return data_fidelity + lambda_reg * regularization

def gradient_function(z_flat, s_dx, s_dy, lambda_reg):
    z = z_flat.reshape(s_dx.shape)
    dz_dx = np.gradient(z, axis=1)
    dz_dy = np.gradient(z, axis=0)
    grad_data_fidelity_x = 2 * (dz_dx - s_dx)
    grad_data_fidelity_y = 2 * (dz_dy - s_dy)
    laplacian = np.gradient(np.gradient(z, axis=1), axis=1) + np.gradient(np.gradient(z, axis=0), axis=0)
    grad_regularization = 2 * lambda_reg * laplacian
    grad_total = np.gradient(grad_data_fidelity_x, axis=1) + np.gradient(grad_data_fidelity_y, axis=0) + grad_regularization
    return grad_total.flatten()

# Test the gradient calculation with a simple case
z_test = np.random.rand(10, 10)
s_dx_test, s_dy_test = np.gradient(z_test)
z_flat_test = z_test.flatten()
lambda_reg_test = 0.1

from scipy.optimize import check_grad
error = check_grad(lambda z: energy_function(z, s_dx_test, s_dy_test, lambda_reg_test),
                   lambda z: gradient_function(z, s_dx_test, s_dy_test, lambda_reg_test),
                   z_flat_test)
print("Gradient check error:", error)