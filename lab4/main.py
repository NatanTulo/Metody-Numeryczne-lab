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
# Dodanie punktów P1, P2, P3, P4 do wykresu:
points = [(-1.375, 1.080), (-0.5339, 4.213), (-0.5011, 4.25), (0.90848, -0.293)]
for p in points:
    plt.plot(p[0], p[1], 'ro')  # Zaznaczenie punktu czerwoną kropką
plt.show()

# przybliżone rozwiązania graficzne (na oko) - 4 rozwiązania:
# P1(-1.375,1.080)
# P2(-0.5339,4.213)
# P3(-0.5011,4.25)
# P4(0.90848,-0.293)


