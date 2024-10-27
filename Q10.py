import numpy as np

def f(x):
    return 3*x - 2*x**2 + x**3 + 2*x**4

def f_dash(x):
    return 3 - 4*x + 3*x**2 + 8*x**3

def f_double_dash(x):
    return -4 + 6*x + 24*x**2

def bisection_search(a, b, tol=1e-5, max_iter=100):
    if a < 0:
        a = 0
    if b < 0:
        return None
    
    iter_count = 0
    while (b - a) > tol and iter_count < max_iter:
        mid = (a + b) / 2
        if f_dash(mid) == 0:
            return mid
        elif f_dash(mid) * f_dash(a) < 0:
            b = mid
        else:
            a = mid
        iter_count += 1
        
    return (a + b) / 2 if iter_count < max_iter else None

def newton_method(x0, tol=1e-5, max_iter=100):
    x = max(x0, 0)
    iter_count = 0
    
    while iter_count < max_iter:
        x_new = x - f_dash(x) / f_double_dash(x)
        x_new = max(x_new, 0)  # Ensure x_new is non-negative
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
        iter_count += 1
    
    return None 

def fibonacci_search(a, b, tol=1e-5, max_iter=100):
    if a < 0:
        a = 0
    if b < 0: 
        return None 

    fib = [0, 1]
    while len(fib) < max_iter + 2:
        fib.append(fib[-1] + fib[-2])
    
    n = len(fib) - 2
    if n < 2:
        return None 

    x1 = a + fib[n - 2] / fib[n] * (b - a)
    x2 = a + fib[n - 1] / fib[n] * (b - a)

    iter_count = 0
    while iter_count < max_iter:
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        
        if abs(b - a) <= tol:
            break

        iter_count += 1

        if n - iter_count >= 2:
            x1 = a + fib[n - iter_count - 2] / fib[n - iter_count] * (b - a)
            x2 = a + fib[n - iter_count - 1] / fib[n - iter_count] * (b - a)
        else:
            break  

    return (a + b) / 2 if iter_count < max_iter else None

x_start = 6
tol = 1e-5
a, b = 0, x_start 

bisection_result = bisection_search(a, b)
newton_result = newton_method(x_start)
fibonacci_result = fibonacci_search(a, b)