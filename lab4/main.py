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
plt.legend()
plt.grid(True)
plt.show()

# przybliżone rozwiązania graficzne (na oko) - 4 rozwiązania:
# P1(-0.00066,1.080)
# P2(-0.5339,4.213)
# P3(-0.5011,4.25)
# P4(0.90848,-0.293)

def iterative_substitution(x1_init, x2_init, tol=1e-6, max_iter=100):
    x1, x2 = x1_init, x2_init
    for _ in range(max_iter):
        x1_new = f1(x2)
        x2_new = f2(x1_new)
        
        if abs(x1_new - x1) < tol and abs(x2_new - x2) < tol:
            return x1_new, x2_new
        
        x1, x2 = x1_new, x2_new
    
    return x1, x2  # Zwracamy najlepsze przybliżenie

# Przykładowe wywołanie
x1_sol, x2_sol = iterative_substitution(1.5, 3.5)
print(f"Rozwiązanie: x1 = {x1_sol}, x2 = {x2_sol}")
