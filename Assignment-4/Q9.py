import numpy as np

def f(x):
    return 32 * x[0]**2 + 12 * x[1]**2 - x[0] * x[1] - 2 * x[0]

def gradient(x):
    dfdx1 = 64 * x[0] - x[1] - 2
    dfdx2 = 24 * x[1] - x[0]
    return np.array([dfdx1, dfdx2])

def hessian(x):
    d2fdx1dx1 = 64
    d2fdx1dx2 = -1
    d2fdx2dx2 = 24
    return np.array([[d2fdx1dx1, d2fdx1dx2], [d2fdx1dx2, d2fdx2dx2]])

def steepest_descent(initial_point, learning_rate=0.01, tolerance=1e-6, max_iter=1000):
    x = np.array(initial_point)
    for i in range(max_iter):
        grad = gradient(x)
        x_new = x - learning_rate * grad
        if np.linalg.norm(x_new - x) < tolerance:
            # print(f"Converged in {i} iterations.")
            break
        x = x_new
    return x

def newtons_method(initial_point, tolerance=1e-6, max_iter=1000):
    x = np.array(initial_point)
    for _ in range(max_iter):
        grad = gradient(x)
        hess = hessian(x)
        x_new = x - np.linalg.inv(hess).dot(grad)
        if np.linalg.norm(x_new - x) < tolerance:
            break
        x = x_new
    return x

initial_point = (-2, 4)
optimal_sd = steepest_descent(initial_point)
optimal_nm = newtons_method(initial_point)
