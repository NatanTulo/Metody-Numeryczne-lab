import numpy as np
import matplotlib.pyplot as plt
import sys

def funDerivativeApprox(x, dx, fun):
    return (fun(x + dx) - fun(x - dx)) / (2 * dx)

def fun(x):
    return np.arcsin(x)

# Analiza przybliżenia pochodnej funkcji arcsin w punkcie x = 0.5
x_0 = 0.5
num_steps = 20
initial_dx = 0.4

true_derivative = 1/np.sqrt(1 - x_0**2)  # pochodna arcsin(x): 1/sqrt(1-x^2)

steps = np.arange(num_steps)               # tworzy wektor indeksów
dx_vals = initial_dx / (5**steps)            # wektor wartości dx
approx_derivatives = funDerivativeApprox(x_0, dx_vals, fun)
errors = np.abs(approx_derivatives - true_derivative)

header = "dx                   approx derivative    absolute error"
np.savetxt(sys.stdout, np.column_stack((dx_vals, approx_derivatives, errors)),
           fmt="%.15f    %.15f    %.15f", header=header, comments='')

# Wykres błędu w skali log-log
plt.figure()
plt.loglog(dx_vals, errors, marker='o')
plt.xlabel("dx")
plt.ylabel("absolute error")
plt.title("Absolute error of derivative approximation")
plt.grid(True, which="both", ls="--")
plt.show()

