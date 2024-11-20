# # # # # import numpy as np
# # # # # from scipy.optimize import minimize, LinearConstraint

# # # # # # Define the objective function
# # # # # def objective(x):
# # # # #     x1, x2, x3 = x
# # # # #     return x1**2 + 2*x1*x2 + 3*x2**2 + 4*x1 + 5*x2 + 6*x3

# # # # # # Define the constraints
# # # # # def constraint1(x):
# # # # #     x1, x2 = x[:2]
# # # # #     return x1 + 2*x2 - 3

# # # # # def constraint2(x):
# # # # #     x1, x3 = x[0], x[2]
# # # # #     return 4*x1 + 5*x3 - 6

# # # # # # Initial guess
# # # # # x0 = np.array([0.0, 0.0, 0.0])

# # # # # # Define the constraints in a format suitable for minimize
# # # # # constraints = [{'type': 'eq', 'fun': constraint1},
# # # # #                {'type': 'eq', 'fun': constraint2}]

# # # # # # Perform the optimization
# # # # # result = minimize(objective, x0, constraints=constraints)

# # # # # # Output the results
# # # # # print(f"Optimal values:\n x1 = {result.x[0]:.4f}\n x2 = {result.x[1]:.4f}\n x3 = {result.x[2]:.4f}")
# # # # # print(f"Minimum value of the objective function: {result.fun:.4f}")
# # # # # print(solution)


# # # # import numpy as np
# # # # from scipy.optimize import minimize

# # # # # Define the objective function (negative for maximization)
# # # # def objective(x):
# # # #     x1, x2 = x
# # # #     return -(4 * x1 + x2**2)

# # # # # Define the constraint
# # # # def constraint(x):
# # # #     x1, x2 = x
# # # #     return x1**2 + x2**2 - 9  # x1^2 + x2^2 = 9 => x1^2 + x2^2 - 9 = 0

# # # # # Initial guess
# # # # x0 = np.array([0.0, 0.0])

# # # # # Define the constraint in a format suitable for minimize
# # # # constraints = [{'type': 'eq', 'fun': constraint}]

# # # # # Perform the optimization
# # # # result = minimize(objective, x0, constraints=constraints)

# # # # # Output the results
# # # # print(f"Optimal values:\n x1 = {result.x[0]:.4f}\n x2 = {result.x[1]:.4f}")
# # # # print(f"Maximum value of the objective function: {-result.fun:.4f}")  # Negate the result to get the maximum
# # # # print(result)

# # # import numpy as np
# # # from scipy.optimize import fsolve

# # # # Define the objective function f(x1, x2) = x1^2 + x2^2
# # # def objective(x):
# # #     x1, x2 = x
# # #     return x1**2 + x2**2

# # # # Define the constraint function 8x1^2 + 6x1x2 + 6x2^2 = 200
# # # def constraint(x):
# # #     x1, x2 = x
# # #     return 8*x1**2 + 6*x1*x2 + 6*x2**2 - 200

# # # # Define the system of equations derived from Lagrange multipliers
# # # # Equation (1): 2x1 + lambda(16x1 + 6x2) = 0
# # # # Equation (2): 2x2 + lambda(6x1 + 12x2) = 0
# # # # This solves the system for lambda as a common variable and applies the constraint
# # # def equations(vars):
# # #     x1, x2, lambd = vars
# # #     eq1 = 2*x1 + lambd * (16*x1 + 6*x2)
# # #     eq2 = 2*x2 + lambd * (6*x1 + 12*x2)
# # #     eq3 = 8*x1**2 + 6*x1*x2 + 6*x2**2 - 200
# # #     return [eq1, eq2, eq3]

# # # # Initial guess for x1, x2, and lambda
# # # initial_guess = [1.0, 1.0, 1.0]

# # # # Solve the system of equations using fsolve
# # # solution = fsolve(equations, initial_guess)

# # # # Extract the solutions for x1, x2, and lambda
# # # x1, x2, lambd = solution

# # # # Compute the value of the objective function at the solution point
# # # min_value = objective([x1, x2])

# # # # Print the results
# # # print(f"Optimal x1: {x1:.4f}")
# # # print(f"Optimal x2: {x2:.4f}")
# # # print(f"Lagrange multiplier (lambda): {lambd:.4f}")
# # # print(f"Minimum value of the objective function: {min_value:.4f}")

# # import sympy as sp

# # x1, x2, lambd = sp.symbols('x1 x2 lambd')

# # f = x1**2 + x2**2
# # g = 8*x1**2 + 6*x1*x2 + 6*x2**2 - 200

# # eq1 = sp.Eq(2*x1 + lambd * (16*x1 + 6*x2), 0)
# # eq2 = sp.Eq(2*x2 + lambd * (6*x1 + 12*x2), 0)
# # eq3 = sp.Eq(8*x1**2 + 6*x1*x2 + 6*x2**2, 200)

# # solution = sp.solve([eq1, eq2, eq3], (x1, x2, lambd))

# # for sol in solution:
# #     x1_sol, x2_sol, lambd_sol = sol
# #     print(f"x1 = {sp.simplify(x1_sol)} or {x1_sol:.4f}")
# #     print(f"x2 = {sp.simplify(x2_sol)} or {x2_sol:.4f}")
# #     print(f"lambda = {sp.simplify(lambd_sol)}")
# #     print('-' * 40)

# # for sol in solution:
# #     x1_sol, x2_sol, _ = sol
# #     objective_value = f.subs({x1: x1_sol, x2: x2_sol})
# #     print(f"Objective function value for x1 = {x1_sol:.4f}, x2 = {x2_sol:.4f}: {objective_value} or {objective_value:.4f}")

# # import numpy as np
# # from scipy.optimize import minimize
# # import sympy as sp

# # def objective(x):
# #     return x[0]**2 + 9*x[1]**2

# # def constraint1(x):
# #     return 2 * x[0] + x[1] - 1  # This constraint must be >= 0

# # def constraint2(x):
# #     return x[0] + 3 * x[1] - 1  # This constraint must be >= 0

# # # Initial guess (using floats since the optimizer requires numerical input)
# # x0 = [0.5, 0.5]

# # # Bounds for x1 and x2 (both must be non-negative)
# # bounds = [(0, None), (0, None)]

# # # Constraints in the form required by scipy.optimize
# # constraints = [{'type': 'ineq', 'fun': constraint1},
# #                {'type': 'ineq', 'fun': constraint2}]

# # # Minimize the objective function
# # solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

# # # Display the results
# # if solution.success:
# #     # Convert to sympy Rational for exact fractions
# #     x1_exact = sp.Rational(solution.x[0]).limit_denominator()
# #     x2_exact = sp.Rational(solution.x[1]).limit_denominator()
    
# #     print('Optimal value:', solution.fun)
# #     print(f"x1 = {x1_exact} or {x1_exact.evalf()}")
# #     print(f"x2 = {x2_exact} or {x2_exact.evalf()}")
# # else:
# #     print("Optimization failed:", solution.message)

# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Define the function f(x, y)
# # def f(x, y):
# #     # Calculate f(x, y) where x^2 + y^2 <= 9
# #     valid_mask = (x**2 + y**2) <= 9
# #     result = np.zeros_like(x)  # Create an array to hold results
# #     result[valid_mask] = x[valid_mask] * y[valid_mask] + np.sqrt(9 - x[valid_mask]**2 - y[valid_mask]**2)
# #     result[~valid_mask] = -np.inf  # Assign -inf to points outside the circle
# #     return result

# # # Create a grid of points in the region x^2 + y^2 <= 9
# # x_values = np.linspace(-3, 3, 10000)
# # y_values = np.linspace(-3, 3, 10000)
# # X, Y = np.meshgrid(x_values, y_values)

# # # Evaluate f(x, y) on the grid
# # Z = f(X, Y)

# # # Find maximum and minimum values
# # max_value = np.max(Z)
# # min_value = np.min(Z)

# # # Print the results
# # print(f'Maximum value of f(x, y): {max_value}')
# # print(f'Minimum value of f(x, y): {min_value}')

# # # Optionally, plot the function over the region
# # plt.figure(figsize=(8, 6))
# # plt.contourf(X, Y, Z, levels=50, cmap='viridis')
# # plt.colorbar(label='f(x, y)')
# # plt.title('Contour plot of f(x, y) in the region x² + y² ≤ 9')
# # plt.xlabel('x')
# # plt.ylabel('y')
# # plt.xlim(-3, 3)
# # plt.ylim(-3, 3)
# # plt.axhline(0, color='black', linewidth=0.5, ls='--')
# # plt.axvline(0, color='black', linewidth=0.5, ls='--')
# # plt.gca().set_aspect('equal', adjustable='box')
# # plt.savefig('Q5 (b).png')
# # plt.show()

# import numpy as np
# from scipy.optimize import minimize

# # Define the objective function: 5x^2 + 6y^2
# def objective(vars):
#     x, y = vars
#     return 5 * x**2 + 6 * y**2

# # Define the inequality constraint: x^2 + y^2 >= 25 (equivalent to x^2 + y^2 - 25 >= 0)
# def constraint1(vars):
#     x, y = vars
#     return x**2 + y**2 - 25

# # Define the inequality constraint: x <= 4 (equivalent to 4 - x >= 0)
# def constraint2(vars):
#     x, y = vars
#     return 4 - x

# # Initial guess for the variables [x, y]
# x0 = [-1, 0]

# # Define the constraints in the form expected by scipy.optimize
# cons = [{'type': 'ineq', 'fun': constraint1},  # x^2 + y^2 >= 25
#         {'type': 'ineq', 'fun': constraint2}]  # x <= 4

# # Solve the problem using minimize
# solution = minimize(objective, x0, method='SLSQP', constraints=cons)

# # Print the result
# print('Optimal solution (x, y):', solution.x)
# print('Objective function value:', solution.fun)

import numpy as np
from scipy.optimize import minimize
from fractions import Fraction

# Define the objective function
def objective(vars):
    x, y = vars
    return (x - 3/2)**2 + (y - 1)**4

# Define the constraint
def constraint(vars):
    x, y = vars
    return x + y - 1  # Should be equal to 0

# Initial guess (x, y)
initial_guess = [0.5, 0.5]

# Define the constraints in the form required by `minimize`
constraints = {'type': 'eq', 'fun': constraint}

# Run the optimization
result = minimize(objective, initial_guess, constraints=constraints)

# Output the results in fractional form
optimal_x = Fraction(result.x[0]).limit_denominator()
optimal_y = Fraction(result.x[1]).limit_denominator()
minimum_value = Fraction(result.fun).limit_denominator()

print(f"Optimal x: {optimal_x}")
print(f"Optimal y: {optimal_y}")
print(f"Minimum value: {minimum_value}")

