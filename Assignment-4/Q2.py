import numpy as np

def f(x):
    return x**2 + 4 * np.cos(x)

def f_dash(x):
    return 2 * x - 4 * np.sin(x)

def f_double_dash(x):
    return 2 - 4 * np.cos(x)

x = 1.5
iterations = 4

for i in range(iterations):
    x = x - f_dash(x) / f_double_dash(x)
    x = max(1, min(x, 2))
    print(f"Iteration {i + 1}: x = {x}, f(x) = {f(x)}")