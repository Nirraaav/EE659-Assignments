import numpy as np

def f(x):
    x1, x2 = x
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

def grad_f(x):
    x1, x2 = x
    df_dx1 = -400 * x1 * (x2 - x1**2) - 2 * (1 - x1)
    df_dx2 = 200 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2])

def backtracking_line_search(x, p, alpha=1, rho=0.9, c=1e-4):
    while f(x + alpha * p) > f(x) + c * alpha * np.dot(grad_f(x), p):
        alpha *= rho
    return alpha

def steepest_descent(x0, tol=1e-4, max_iter=1000):
    x = x0
    alpha = 1  # Initial step length
    iterations = []

    for i in range(max_iter):
        gradient = grad_f(x)
        if np.linalg.norm(gradient) < tol:
            break

        p = -gradient
        alpha = backtracking_line_search(x, p) 
        x = x + alpha * p 

        iterations.append((i + 1, x, alpha)) 

    return x, iterations

x0 = np.array([1.2, 1.2])
result, iterations = steepest_descent(x0)

print(f"Optimal point: {result}")
print("Iterations (index, x, step length):")
for iteration in iterations:
    print(iteration)
