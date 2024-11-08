import numpy as np

def f(x):
    return 40 * x**8 - 15 * x**7 + 70 * x**6 - 10 * x**5 + 20 * x**4 - 14 * x**3 + 60 * x**2 - 70 * x

def f_dash(x):
    return 320 * x**7 - 105 * x**6 + 420 * x**5 - 50 * x**4 + 80 * x**3 - 42 * x**2 + 120 * x - 70

def bisection_method(a, b, tol=0.01):
    while (b - a) / 2 > tol:
        mid = (a + b) / 2
        if f_dash(mid) == 0:
            return mid
        elif f_dash(mid) > 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2

def golden_section_method(a, b, tol=0.005):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2

a, b = -1, 1

x_min_bisection = bisection_method(a, b, tol=0.01)
f_min_bisection = f(x_min_bisection)

x_min_golden = golden_section_method(a, b, tol=0.005)
f_min_golden = f(x_min_golden)

print(x_min_bisection, x_min_golden)