import numpy as np

def f(x):
    x1, x2 = x
    return (3 - x1)**2 + 7 * (x2 - x1**2)**2

def grad_f(x):
    x1, x2 = x
    df_dx1 = -2 * (3 - x1) - 14 * (x2 - x1**2) * 2 * x1
    df_dx2 = 14 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2])

def dfp_method(x0, tol=1e-6, max_iter=1000):
    x = x0
    grad = grad_f(x)
    H = np.eye(len(x))
    iter_count = 0
    
    while np.linalg.norm(grad) > tol and iter_count < max_iter:
        p = -np.dot(H, grad)  # search direction
        alpha = 0.01  
        x_new = x + alpha * p
        grad_new = grad_f(x_new)
        
        s = x_new - x
        y = grad_new - grad
        
        H = H + np.outer(y, y) / np.dot(y, s) - np.dot(np.dot(H, np.outer(s, s)), H) / np.dot(s, np.dot(H, s))
        
        x = x_new
        grad = grad_new
        iter_count += 1
    
    return x, iter_count

x0 = np.array([0.0, 0.0])

solution_dfp, iterations_dfp = dfp_method(x0)
print(f"Davidon-Fletcher-Powell solution: {solution_dfp}")
print(f"Iterations: {iterations_dfp}")