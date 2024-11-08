import numpy as np

def f(x):
    return 6 * np.exp(-2 * x) + 2 * x**2

def golden_section_search(func, a, b, tol=1e-5):
    gr = (np.sqrt(5) + 1) / 2

    c = b - (b - a) / gr
    d = a + (b - a) / gr

    while abs(b - a) > tol:
        if func(c) < func(d):
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2

def dichotomous_search(func, a, b, tol=1e-5, delta=1e-6):
    while abs(b - a) > tol:
        mid = (a + b) / 2
        x1 = mid - delta
        x2 = mid + delta

        if func(x1) < func(x2):
            b = x2
        else:
            a = x1

    return (a + b) / 2

a, b = -10, 10

min_golden = golden_section_search(f, a, b)
min_dichotomous = dichotomous_search(f, a, b)
val_golden = f(min_golden)
val_dichotomous = f(min_dichotomous)
print(f"Golden section search: {min_golden}, {val_golden}")
print(f"Dichotomous search: {min_dichotomous}, {val_dichotomous}")