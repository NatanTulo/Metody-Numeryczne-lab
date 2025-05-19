import numpy as np
import math

def f(x):
    """Define the function to integrate."""
    return -0.03654 * x**4 + 0.7655 * x**3 + 0.543 * x**2 + 2.663 * x + 1.543

def integrate_analytically(a, b):
    """
    Analytically integrate the function and return the result.
    Integrate each term separately and sum them.
    """
    def integrate_term(coeff, power):
        """Integrate a single term of the polynomial."""
        return coeff * (b**(power+1) - a**(power+1)) / (power + 1)

    terms = [
        (-0.03654, 4),
        (0.7655, 3),
        (0.543, 2),
        (2.663, 1),
        (1.543, 0)
    ]

    return sum(integrate_term(coeff, power) for coeff, power in terms)

def trapezoidal_method(a, b, n):
    """
    Numerical integration using the trapezoidal rule with n segments.
    """
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    integral = h/2 * (y[0] + 2*np.sum(y[1:-1]) + y[-1])

    def f_double_prime(x):
        return -0.14616 * x**2 + 2.2965 * x + 1.086

    mean_f_double_prime = np.mean(f_double_prime(np.linspace(a, b, 100)))
    error_estimate = -mean_f_double_prime * (b-a)**3 / (12 * n**2)

    return integral, abs(error_estimate)

def romberg_integration(a, b, max_iterations=10, error_threshold=0.002):
    """
    Romberg integration method with error threshold.
    """
    R = np.zeros((max_iterations, max_iterations))

    for i in range(max_iterations):
        n = 2**i
        R[i, 0], _ = trapezoidal_method(a, b, n)

    for i in range(1, max_iterations):
        for j in range(1, i+1):
            R[i, j] = (4**j * R[i, j-1] - R[i-1, j-1]) / (4**j - 1)

    for i in range(1, max_iterations):
        for j in range(1, i+1):
            relative_error = abs((R[i, j] - R[i, j-1]) / R[i, j]) * 100

            if relative_error < error_threshold:
                return R[i, j], R[:i+1, :j+1], relative_error

    raise ValueError("Could not achieve desired error threshold")

def gauss_quadrature_3pt(a, b):
    """
    3-point Gauss-Legendre quadrature.
    """
    def transform(t):
        return 0.5 * (b-a) * t + 0.5 * (b+a)

    weights = [5/9, 8/9, 5/9]
    points = [-math.sqrt(3/5), 0, math.sqrt(3/5)]

    integral = 0.5 * (b-a) * sum(
        w * f(transform(t)) for w, t in zip(weights, points)
    )

    return integral

a, b = -8, 23

analytical_result = integrate_analytically(a, b)
print("Analytical Integration:")
print(f"Result: {analytical_result}")

print("\nTrapezoidal Method Results:")
print("n\tIntegral\t\tAbs Error\t\tRel Error (%)")
trapezoidal_results = []
for n in range(2, 11):
    integral, error = trapezoidal_method(a, b, n)
    rel_error = abs((integral - analytical_result) / analytical_result) * 100
    print(f"{n}\t{integral:.6f}\t\t{abs(integral-analytical_result):.6f}\t\t{rel_error:.6f}")
    trapezoidal_results.append((n, integral, abs(integral-analytical_result), rel_error))

print("\nRomberg Integration:")
romberg_result, romberg_table, romberg_error = romberg_integration(a, b)
print(f"Result: {romberg_result}")
print(f"Relative Error: {romberg_error:.6f}%")

gauss_result = gauss_quadrature_3pt(a, b)
print("\n3-point Gauss Quadrature:")
print(f"Result: {gauss_result}")
print(f"Absolute Error: {abs(gauss_result - analytical_result)}")
print(f"Relative Error: {abs(gauss_result - analytical_result) / abs(analytical_result) * 100:.6f}%")

print("\nRomberg Integration Table:")
print(romberg_table)