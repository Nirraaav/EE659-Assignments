import numpy as np

def f(x):
    return (np.sin(x))**6 * np.tan(1 - x) * np.exp(30 * x)

def golden_ratio_method(a, b, tol, max_iter):
    phi = (1 + np.sqrt(5)) / 2
    res_phi = 2 - phi
    
    x1 = b - res_phi * (b - a)
    x2 = a + res_phi * (b - a)
    
    f1 = f(x1)
    f2 = f(x2)

    iterations = 0
    while (b - a) > tol and iterations < max_iter:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - res_phi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + res_phi * (b - a)
            f2 = f(x2)
        iterations += 1

    return (a + b) / 2, f((a + b) / 2)

def quadratic_interpolation_method(a, b, tol, max_iter):
    x0 = a
    x1 = (a + b) / 2
    x2 = b

    iterations = 0
    while (b - a) > tol and iterations < max_iter:
        f0, f1, f2 = f(x0), f(x1), f(x2)
        denominator = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if denominator == 0:
            break  # Avoid division by zero
        x_new = (f0 * (x1 - x2) + f1 * (x2 - x0) + f2 * (x0 - x1)) / (f0 * (x1 - x2) + f1 * (x2 - x0) + f2 * (x0 - x1))
        
        if x_new < a or x_new > b:
            break  # Ensure new point is within bounds

        if f(x_new) > f(x1):
            x0, x1, x2 = x1, x_new, x2
        else:
            x0, x1, x2 = x0, x1, x_new

        iterations += 1

    return (x0 + x1 + x2) / 3, f((x0 + x1 + x2) / 3)

def f_prime(x):  # Numerical derivative using central difference (because im too lazy to calculate the real derivative and I dont want to use the scipy function)
    h = 1e-5
    return (f(x + h) - f(x - h)) / (2 * h)

def goldstein_line_search(x0, direction, alpha=0.1, beta=0.9, max_iter=100):
    x1 = x0 + direction
    iterations = 0

    while (f(x1) > f(x0) + alpha * (x1 - x0) * f_prime(x0) and
           f(x1) < f(x0) + beta * (x1 - x0) * f_prime(x0)) and iterations < max_iter:
        x1 -= direction
        iterations += 1
    
    return x1, f(x1)

a, b = 0, 1
tolerance = 0.15
max_iterations = 1000

optimum_golden = golden_ratio_method(a, b, tolerance, max_iterations)
optimum_quad = quadratic_interpolation_method(a, b, tolerance, max_iterations)

x0 = 0.5  # starting point
direction = 0.1  # arbitrary small step
optimum_goldstein = goldstein_line_search(x0, direction, max_iter=max_iterations)

optimum_golden = golden_ratio_method(a, b, tolerance, max_iterations)
print("Golden Ratio Method: x =", optimum_golden[0], ", f(x) =", optimum_golden[1])

# Quadratic Interpolation Method
optimum_quad = quadratic_interpolation_method(a, b, tolerance, max_iterations)
print("Quadratic Interpolation Method: x =", optimum_quad[0], ", f(x) =", optimum_quad[1])

# Goldstein Line Search
x0 = 0.5  # starting point
direction = 0.1  # arbitrary small step
optimum_goldstein = goldstein_line_search(x0, direction, max_iter=max_iterations)
print("Goldstein Line Search: x =", optimum_goldstein[0], ", f(x) =", optimum_goldstein[1])