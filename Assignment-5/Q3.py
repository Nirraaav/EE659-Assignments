import numpy as np
from scipy.linalg import solve

A = np.array([
    [2, 0, 0, 0, 2, 1],
    [0, 2, 0, 0, 1, 1],
    [0, 0, 2, 0, 1, 3],
    [0, 0, 0, 2, 4, 1],
    [2, 1, 1, 4, 0, 0],
    [1, 1, 3, 1, 0, 0]
])

b = np.array([2, 0, 0, 3, 7, 6])

solution = solve(A, b)

x1, x2, x3, x4, lambda1, lambda2 = solution
print("Solution:")
print(f"x1 = {x1}, x2 = {x2}, x3 = {x3}, x4 = {x4}, lambda1 = {lambda1}, lambda2 = {lambda2}")
