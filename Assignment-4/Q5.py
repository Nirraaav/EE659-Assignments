import numpy

def f(x):
    return 2 * x**2 - 5 * x

def f_dash(x):
    return 4 * x - 5

def newton_method(initial_guess, tolerance=1e-7, max_iterations=1000):
    x = initial_guess
    for iteration in range(max_iterations):
        fx = f(x)
        derivative = f_dash(x)
        
        if derivative == 0:
            return None
        
        x = x - fx / derivative
        
        print(f"Iteration {iteration + 1}: x = {x}, f(x) = {f(x)}")
        
        if abs(f(x)) < tolerance:
            return x
    
    return None

initial_guess = 5
newton_method(initial_guess)
