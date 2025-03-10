import numpy as np
import matplotlib.pyplot as plt

def funDerivativeApprox(x, dx, fun):
    return (fun(x + dx) - fun(x - dx)) / (2 * dx)

def fun(x):
    return np.arcsin(x)

# Analiza przybliżenia pochodnej funkcji arcsin w punkcie x = 0.5
x_0 = 0.5
num_steps = 20
initial_dx = 0.4

print("dx                     approx derivative")
for i in range(num_steps):
    dx = initial_dx / (5**i)
    approx_derivative = funDerivativeApprox(x_0, dx, fun)
    print(f"{dx:.20f} {approx_derivative:.10f}")

