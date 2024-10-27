import numpy as np

def f(t):
    return np.exp(-t) + np.exp(t)

def golden_section_method(a, b, tol=0.15, max_iter=100):
    phi = (1 + np.sqrt(5)) / 2
    iter_count = 0
    while (b - a) > tol and iter_count < max_iter:
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        if f(c) < f(d):
            b = d
        else:
            a = c
        iter_count += 1
    return (a + b) / 2

def fibonacci_method(a, b, tol=0.15, max_iter=100):
    n = 1
    while (b - a) > tol and n < max_iter:
        n += 1

    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[-1] + fib[-2])

    for i in range(1, n):
        r1 = a + (fib[n - i - 1] / fib[n]) * (b - a)
        r2 = a + (fib[n - i] / fib[n]) * (b - a)

        if f(r1) < f(r2):
            b = r2
        else:
            a = r1

    return (a + b) / 2

def armijo_line_search(t0, alpha=0.1, beta=0.5, tol=0.15, max_iter=100):
    t = t0
    iter_count = 0
    while iter_count < max_iter:
        gradient = -np.exp(-t) + np.exp(t)  # f'(t)
        t_new = t - alpha * gradient
        if f(t_new) <= f(t) + beta * alpha * gradient:
            t = t_new
        if abs(f(t_new) - f(t)) < tol:
            break
        iter_count += 1
    return t

a, b = -1, 1
tol = 0.15
max_iter = 10000

golden_result = golden_section_method(a, b, tol, max_iter)
fibonacci_result = fibonacci_method(a, b, tol, max_iter)
armijo_result = armijo_line_search(0, tol=tol, max_iter=max_iter)

print(golden_result, fibonacci_result, armijo_result)