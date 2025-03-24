import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def f1(x):
    return -3*x*x-2*x+4

# y=-x^2-2xy
# y+2xy=-x^2
# y(1+2x)=-x^2
# y=-x^2/(1+2x)

def f2(x):
    return -x*x/(1+2*x)

x = np.linspace(-2, 2, 1000)
y1 = f1(x)
y2 = f2(x)
plt.plot(x, y1, label='f1(x)')
plt.plot(x, y2, label='f2(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', adjustable='datalim')
plt.xlim(-5, 5)          # dodane ograniczenia osi x
plt.ylim(-10, 5)         # dodane ograniczenia osi y
plt.legend()
plt.grid(True)
# Dodanie punktów P1, P2, P3, P4 do wykresu:
points = [(-1.375, 1.080), (-0.5339, 4.213), (0.90848, -0.293)]
for p in points:
    plt.plot(p[0], p[1], 'ro')  # Zaznaczenie punktu czerwoną kropką
plt.show()

# przybliżone rozwiązania graficzne (na oko) - 3 rozwiązania:
# P1(-1.375,1.080)
# P2(-0.5339,4.213)
# P4(0.90848,-0.293)


def iterative_substitution(initial_guess, max_iterations=1000, tolerance=1e-10):
    """
    Implements the iterative substitution method to find an intersection point
    between f1(x) and f2(x).
    """
    x = initial_guess
    iterations = []
    
    # Use a damping factor alpha to help with convergence
    alpha = 0.05
    
    for i in range(max_iterations):
        iterations.append(x)
        
        # Calculate new x value using the iterative formula
        x_new = x + alpha * (f2(x) - f1(x))
        
        # Check if we've converged
        if abs(x_new - x) < tolerance:
            return x_new, iterations, True
        
        x = x_new
    
    return x, iterations, False  # Failed to converge

# Find a single solution using an initial guess
initial_guess = -1.4  # Using a value close to P1 from previous observations
solution, iterations, converged = iterative_substitution(initial_guess)

if converged:
    y_sol = f1(solution)  # or f2(solution), they should be approximately equal
    print(f"Solution found using iterative substitution method:")
    print(f"x = {solution:.10f}")
    print(f"y = {y_sol:.10f}")
    print(f"Error: |f1(x) - f2(x)| = {abs(f1(solution) - f2(solution)):.10f}")
    print(f"Iterations required: {len(iterations)}")
else:
    print("Failed to converge")

# Plot the functions and the solution
x = np.linspace(-2, 2, 1000)
y1 = f1(x)
y2 = f2(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='f1(x) = -3x²-2x+4')
plt.plot(x, y2, label='f2(x) = -x²/(1+2x)')
plt.plot(solution, y_sol, 'ro', label='Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='datalim')
plt.xlim(-5, 5)          # dodane ograniczenia osi x
plt.ylim(-10, 5)         # dodane ograniczenia osi y
plt.legend()
plt.title('Intersection Point Using Iterative Substitution Method')
plt.show()

# Dodanie rozwiązania i wykresu metodą bisekcji

def f(x):
    return f1(x) - f2(x)

bisect_solution = opt.bisect(f, -1.4, -1.36, xtol=1e-10)
y_bisect = f1(bisect_solution)
print("\nRozwiązanie znalezione metodą bisekcji:")
print(f"x = {bisect_solution:.10f}")
print(f"y = {y_bisect:.10f}")
print(f"Error: |f1(x) - f2(x)| = {abs(f1(bisect_solution)-f2(bisect_solution)):.10f}")

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='f1(x) = -3x²-2x+4')
plt.plot(x, y2, label='f2(x) = -x²/(1+2x)')
plt.plot(bisect_solution, y_bisect, 'go', label='Rozwiązanie bisekcją')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='datalim')
plt.xlim(-5, 5)          # dodane ograniczenia osi x
plt.ylim(-10, 5)         # dodane ograniczenia osi y
plt.legend()
plt.title('Intersection Point Using Bisection Method')
plt.show()

# Dodanie rozwiązania metodą Newtona-Raphsona
# Zmodyfikowano blok metodą Newtona-Raphsona (scipy.optimize.newton)
# Poprzednia wersja:
# initial_guesses = [-1.4, -0.55, -0.5, 0.9]
# Nowa wersja: usunięto punkt -0.5 (trzeci element), który jest niepoprawny
initial_guesses = [-1.4, -0.55, 0.9]
newton_solutions = []

for guess in initial_guesses:
    try:
        sol = opt.newton(f, guess, tol=1e-10, maxiter=1000)
        newton_solutions.append(sol)
    except RuntimeError as e:
        print(f"Newton method failed for initial guess {guess}: {e}")

print("\nRozwiązania znalezione metodą Newtona-Raphsona:")
for sol in newton_solutions:
    y_newton = f1(sol)
    print(f"x = {sol:.10f}, y = {y_newton:.10f}, Error: |f1(x)-f2(x)| = {abs(f1(sol)-f2(sol)):.10e}")

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='f1(x) = -3x²-2x+4')
plt.plot(x, y2, label='f2(x) = -x²/(1+2x)')
for idx, sol in enumerate(newton_solutions):
    y_val = f1(sol)
    plt.plot(sol, y_val, 'mo', label='Newtona-Raphson' if idx == 0 else "")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='datalim')
plt.xlim(-5, 5)          # dodane ograniczenia osi x
plt.ylim(-10, 5)         # dodane ograniczenia osi y
plt.legend()
plt.title('Punkty przecięcia metodą Newtona-Raphsona')
plt.show()


# Dodanie implementacji metody Newtona-Raphsona (własna implementacja z użyciem rozwinięcia Taylora do pierwszej pochodnej)

def newton_method(f, fprime, x0, tol=1e-10, max_iter=1000):
    x = x0
    for i in range(max_iter):
        try:
            fx = f(x)
        except ZeroDivisionError:
            print(f"ZeroDivisionError encountered in f(x) for x = {x:.10f}")
            return x, i, False
        try:
            fpx = fprime(x)
        except ZeroDivisionError:
            print(f"ZeroDivisionError encountered in fprime(x) for x = {x:.10f}")
            return x, i, False
        if abs(fpx) < 1e-12:
            print(f"Pochodna bliska zero dla x = {x:.10f}. Metoda zatrzymana.")
            return x, i, False
        x_new = x - fx/fpx
        if abs(x_new - x) < tol:
            return x_new, i+1, True
        x = x_new
    return x, max_iter, False

def fprime(x):
    # f(x) = f1(x) - f2(x)
    # f1(x) = -3x² - 2x + 4       -> f1'(x) = -6x - 2
    # f2(x) = -x²/(1+2x)          -> f2'(x) = -2x/(1+2x) + 2x²/(1+2x)²
    return (-6*x - 2) + (2*x/(1+2*x) - 2*x*x/(1+2*x)**2)

# Zmodyfikowano blok własnej implementacji metody Newtona-Raphsona
# Poprzednia wersja:
# initial_guesses_own = [-1.4, -0.55, -0.49, 0.9]
# Nowa wersja: usunięto punkt -0.49 (trzeci element), który powoduje błąd (asymptota)
initial_guesses_own = [-1.4, -0.55, 0.9]
own_newton_solutions = []

print("\nRozwiązania znalezione metodą Newtona-Raphsona (własna implementacja):")
for guess in initial_guesses_own:
    sol, iterations, converged = newton_method(f, fprime, guess)
    if converged:
        y_val = f1(sol)
        print(f"x = {sol:.10f}, y = {y_val:.10f}, Iteracje: {iterations}, Error: |f1(x)-f2(x)| = {abs(f1(sol)-f2(sol)):.10e}")
        own_newton_solutions.append(sol)
    else:
        print(f"Metoda nie zbiega dla punktu startowego {guess}")

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='f1(x) = -3x²-2x+4')
plt.plot(x, y2, label='f2(x) = -x²/(1+2x)')
for idx, sol in enumerate(own_newton_solutions):
    y_val = f1(sol)
    plt.plot(sol, y_val, 'co', label='Własna Newtona-Raphson' if idx == 0 else "")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='datalim')
plt.xlim(-5, 5)          # dodane ograniczenia osi x
plt.ylim(-10, 5)         # dodane ograniczenia osi y
plt.legend()
plt.title('Punkty przecięcia metodą Newtona-Raphsona (własna implementacja)')
plt.show()