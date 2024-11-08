import numpy as np
from scipy.optimize import minimize

def objective(x):
    x1, x2 = x
    exp_term = np.exp(np.clip(2 * x1**2 + x2**2, -50, 50))
    return 2 * x1 + 3 * x2**2 + exp_term

def gradient(x):
    x1, x2 = x
    exp_term = np.exp(np.clip(2 * x1**2 + x2**2, -50, 50)) 
    grad_x1 = 2 + 4 * x1 * exp_term
    grad_x2 = 6 * x2 + 2 * x2 * exp_term
    return np.array([grad_x1, grad_x2])

def fletcher_reeves_cg(func, grad, x0, tol=1e-6, max_iter=100):
    x = x0
    g = grad(x)  
    d = -g  
    for i in range(max_iter):
        g = grad(x)
        
        step_size = 0.1 
        x_new = x + step_size * d
        g_new = grad(x_new)
        
        if np.linalg.norm(g_new) < tol:
            print(f"Converged at iteration {i}")
            break
        
        beta = np.dot(g_new, g_new) / np.dot(g, g)
        
        d = -g_new + beta * d
        
        x = x_new
        g = g_new
        
    return x, func(x)

def bfgs(func, grad, x0, tol=1e-6, max_iter=50, epsilon=1e-8):
    x = x0
    n = len(x)
    
    H = np.eye(n)
    
    f = func(x)
    g = grad(x)
    
    for _ in range(max_iter):
        p = -np.dot(H, g)
        
        alpha = 1.0
        x_new = x + alpha * p
        f_new = func(x_new)
        g_new = grad(x_new)
        
        if np.linalg.norm(g_new) < tol:
            return x_new, f_new
        
        s = x_new - x
        y = g_new - g
        
        rho = np.dot(y, s)
        if abs(rho) < epsilon:  # Small value check
            print("Warning: Small curvature. Skipping Hessian update.")
            return x_new, f_new

        rho = 1.0 / rho 

        H = np.dot(np.eye(n) - rho * np.outer(s, y), np.dot(H, np.eye(n) - rho * np.outer(y, s))) + rho * np.outer(s, s)
        
        x = x_new
        f = f_new
        g = g_new
    
    return x, f  

x0 = np.array([0.0, 0.0])

x_opt_cg, f_opt_cg = fletcher_reeves_cg(objective, gradient, x0)
print(f"Optimal Solution: {x_opt_cg}")
print(f"Optimal Objective Value: {f_opt_cg}")

x_opt_bfgs, f_opt_bfgs = bfgs(objective, gradient, x0)
print(f"Optimal Solution: {x_opt_bfgs}")
print(f"Optimal Objective Value: {f_opt_bfgs}")
