import numpy as np
from scipy.optimize import minimize

def objective_function(x):
    x1, x2 = x
    return (3 - x1)**2 + 7 * (x2 - x1**2)**2

def gradient(x):
    x1, x2 = x
    df_dx1 = -2 * (3 - x1) - 28 * x1 * (x2 - x1**2)
    df_dx2 = 14 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2])

initial_point = np.array([0, 0])

def cyclic_coordinate_method(func, x0, tol=1e-6, max_iter=1000, step_size=0.1):
    x = x0.copy()
    n = len(x)
    
    for _ in range(max_iter):
        old_x = x.copy()
        for i in range(n):
            # Line search in the i-th coordinate direction
            while True:
                f_current = func(x)
                
                # Increase x[i] by step_size and evaluate
                x[i] += step_size
                f_new = func(x)
                
                if f_new < f_current:
                    continue
                
                # Decrease x[i] by 2 * step_size and evaluate
                x[i] -= 2 * step_size
                f_new = func(x)
                
                if f_new < f_current:
                    continue
                
                # Reset x[i] to its original value if no improvement
                x[i] += step_size
                break
        
        # Check convergence by comparing the change in x
        if np.linalg.norm(x - old_x) < tol:
            break
    
    return x

def hooke_jeeves(func, x0, step_size=0.5, alpha=2.0, tol=1e-6, max_iter=1000):
    x = x0.copy()
    n = len(x)
    
    def explore(xb, step):
        x_new = xb.copy()
        for i in range(n):
            f_before = func(x_new)
            x_new[i] += step
            if func(x_new) >= f_before:
                x_new[i] -= 2 * step
                if func(x_new) >= f_before:
                    x_new[i] += step
        return x_new
    
    for _ in range(max_iter):
        xb = x.copy()
        xe = explore(x, step_size)
        
        if np.linalg.norm(xe - x) < tol:
            break
        
        x = xe if func(xe) < func(x) else x
        
        xb_new = x + alpha * (x - xb)
        if func(xb_new) < func(x):
            x = xb_new
        else:
            step_size *= 0.5
    
    return x

def rosenbrock_method(func, grad, x0, learning_rate=0.001, tol=1e-6, max_iter=10000):
    x = x0.copy()
    for _ in range(max_iter):
        grad_val = grad(x)
        x_new = x - learning_rate * grad_val
        
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new
    return x

cyclic_result = cyclic_coordinate_method(objective_function, initial_point)
hooke_jeeves_result = hooke_jeeves(objective_function, initial_point)
rosenbrock_result = rosenbrock_method(objective_function, gradient, initial_point)

print(cyclic_result, hooke_jeeves_result, rosenbrock_result)
