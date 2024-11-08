import numpy as np
from scipy.optimize import minimize

def objective(x):
    x1, x2 = x
    return (x1 - 5)**2 + (x2 - 3)**2

def barrier(x, mu):
    x1, x2 = x
    g1 = 6 - (3 * x1 + 2 * x2)
    g2 = 2 - (-4 * x1 + 2 * x2)
    
    if g1 <= 0 or g2 <= 0:
        return np.inf
    
    return -mu * (np.log(g1) + np.log(g2))

def barrier_objective(x, mu):
    return objective(x) + barrier(x, mu)

def solve_barrier_problem(mu, x0):
    result = minimize(barrier_objective, x0, args=(mu), method='CG', options={'disp': True})
    return result.x, result.fun

mu = 1
x0 = np.array([0.0, 0.0]) 

solution, objective_value = solve_barrier_problem(mu, x0)

print("Optimal Solution: ", solution)
print("Optimal Objective Value: ", objective_value)
