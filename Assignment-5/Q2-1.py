import numpy as np

# Define the objective function and its gradient
def f(x):
    x1, x2 = x
    return (3 - x1)**2 + 7 * (x2 - x1**2)**2

def grad_f(x):
    x1, x2 = x
    df_dx1 = -2 * (3 - x1) - 14 * (x2 - x1**2) * 2 * x1
    df_dx2 = 14 * (x2 - x1**2)
    
    grad = np.array([df_dx1, df_dx2])
    grad = np.clip(grad, -10, 10)
    
    return grad

def fletcher_reeves_method(x0, tol=1e-6, max_iter=1000):
    x = x0
    grad = grad_f(x)
    p = grad  
    iter_count = 0
    alpha = 0.01
    
    while np.linalg.norm(grad) > tol and iter_count < max_iter:
        x = x + alpha * p
        new_grad = grad_f(x)
        
        beta = np.dot(new_grad, new_grad) / np.dot(grad, grad)
        p = -new_grad + beta * p
        
        grad = new_grad
        iter_count += 1
        
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            print("Gradient became NaN or Inf!")
            break
    
    return x, iter_count

x0 = np.array([0.0, 0.0])

solution_fletcher_reeves, iterations_fletcher_reeves = fletcher_reeves_method(x0)
print(f"Fletcher and Reeves solution: {solution_fletcher_reeves}")
print(f"Iterations: {iterations_fletcher_reeves}")
