import numpy as np

Q = np.array([[10, 1], [1, 2]])
b = np.array([3, 1])

def conjugate_gradient(Q, b, x0=None, tol=1e-9, max_iter=1000):
    if x0 is None:
        x = np.zeros_like(b)  
    else:
        x = x0
    r = b - np.dot(Q, x)  
    p = r.copy()  
    rs_old = np.dot(r, r)
    
    for i in range(max_iter):
        Ap = np.dot(Q, p)
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        
        if np.sqrt(rs_new) < tol:
            break
        
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    
    return x, i + 1  

x0 = np.zeros(2)

solution, iterations = conjugate_gradient(Q, b, x0)
print(f"Solution: {solution}")
print(f"Iterations: {iterations}")
