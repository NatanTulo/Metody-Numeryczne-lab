import numpy as np
import matplotlib.pyplot as plt
import sys

def funDerivativeApprox(x, dx, fun):
    return (fun(x + dx) - fun(x - dx)) / (2 * dx)

def fun(x):
    return np.arcsin(x)

x_0 = 0.5
num_steps = 20
initial_dx = 0.4

true_derivative = 1/np.sqrt(1 - x_0**2)

steps = np.arange(num_steps)
dx_vals = initial_dx / (5**steps)
approx_derivatives = funDerivativeApprox(x_0, dx_vals, fun)
errors = np.abs(approx_derivatives - true_derivative)

header = "dx                   approx derivative    absolute error"
np.savetxt(sys.stdout, np.column_stack((dx_vals, approx_derivatives, errors)),
           fmt="%.15f    %.15f    %.15f", header=header, comments='')

plt.figure()
plt.loglog(dx_vals, errors, marker='o')
plt.xlabel("dx")
plt.ylabel("absolute error")
plt.title("Absolute error of derivative approximation")
plt.grid(True, which="both", ls="--")
plt.show()

best_index = np.argmin(errors)
best_dx = dx_vals[best_index]
print(f"\nOptimal dx = {best_dx:.10f} with minimal error = {errors[best_index]:.15f}")

x_vals = np.linspace(best_dx, 1 - best_dx, 101)
approx_deriv_x = funDerivativeApprox(x_vals, best_dx, fun)

plt.figure()
plt.plot(x_vals, approx_deriv_x, marker='o', linestyle='-')
plt.xlabel("x")
plt.ylabel("Approximate derivative")
plt.title(f"Approximate derivative using best dx = {best_dx:.10f}")
plt.grid(True)
plt.show()