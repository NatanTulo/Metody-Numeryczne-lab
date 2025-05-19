import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# -----------------------------
# INTERPOLACJA NEWTONA
# -----------------------------

def divided_differences(x, y):
    """
    Oblicza unormowane różnice skończone (współczynniki Newtona).
    x, y: wektory długości n
    Zwraca: wektor coef długości n:
      coef[0] = f[x0]
      coef[1] = f[x1,x0]
      ...
    """
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x[i] - x[i-j])
    return coef


def newton_eval(x_data, coef, x_val):
    """
    Ocena wielomianu Newtona w punktach x_val (Horner dla formy Newtona).
    x_data: węzły, coef: współczynniki, x_val: wartość lub wektor
    """
    x_val = np.array(x_val, ndmin=1)
    n = len(coef)
    result = np.zeros_like(x_val, dtype=float)
    for k in range(n-1, -1, -1):
        result = result * (x_val - x_data[k]) + coef[k]
    return result

# -----------------------------
# CUBIC SPLINE (Scipy)
# -----------------------------

def spline_scipy(x, y, x_new):
    """
    Korzysta z scipy.interpolate.CubicSpline do dopasowania naturalnego splajnu.
    Zwraca wartości w x_new.
    """
    cs = CubicSpline(x, y, bc_type='natural')
    return cs(x_new)

# -----------------------------
# CUBIC SPLINE (własna implementacja)
# -----------------------------

def spline_custom(x, y, x_new):
    """
    Naturalny splajn trzeciego stopnia: c[0]=c[n-1]=0.
    Zwraca wartości interpolowane w x_new.
    """
    n = len(x)
    h = np.diff(x)
    # macierz trójdiagonalna do wyznaczenia drugich pochodnych c
    A = np.zeros((n, n))
    b = np.zeros(n)
    # warunki naturalne
    A[0,0] = 1
    A[-1,-1] = 1
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i]   = 2*(h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3*( (y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1] )
    # rozwiązujemy A c = b
    c = np.linalg.solve(A, b)
    # obliczamy współczynniki a, b_i, d
    a = y[:-1]
    b_coef = (y[1:] - y[:-1]) / h - (2*c[:-1] + c[1:]) * h / 3
    d_coef = (c[1:] - c[:-1]) / (3*h)
    # ocena splajnu w punktach x_new
    y_new = np.zeros_like(x_new, dtype=float)
    for idx, xi in enumerate(x_new):
        # znajdź przedział
        if xi <= x[0]: i = 0
        elif xi >= x[-1]: i = n-2
        else: i = np.searchsorted(x, xi) - 1
        dx = xi - x[i]
        y_new[idx] = a[i] + b_coef[i]*dx + c[i]*dx**2 + d_coef[i]*dx**3
    return y_new

# -----------------------------
# GŁÓWNA FUNKCJA
# -----------------------------

def main():
    # Wczytanie punktów trajektorii robota
    data = np.loadtxt('data9.txt')  # dwie kolumny: x, y
    x = data[:,0]
    y = data[:,1]

    # Generujemy 101 punktów do interpolacji
    x_interp = np.linspace(np.min(x), np.max(x), 101)

    # Newton
    coef = divided_differences(x, y)
    y_newton = newton_eval(x, coef, x_interp)

    # Spline SciPy
    y_scipy = spline_scipy(x, y, x_interp)

    # Spline własny
    y_custom = spline_custom(x, y, x_interp)

    # Wydruk różnic między sygnałami
    print("Różnice sygnału:")
    print("SciPy - wersja z wykładu:", y_scipy - y_custom)
    if( np.max(np.abs(y_scipy - y_custom)) > 1e-10):
        print("Różnice między wykresami są znaczące.")
    else:
        print("Wykresy metod ze SciPy oraz z wykładu są tak bliskie siebie, że nie widać ich na wykresie.")

    # Wykres
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'o', label='Pomiary trajektorii')
    plt.plot(x_interp, y_newton, '-', label='Newton')
    plt.plot(x_interp, y_scipy, '--', label='CubicSpline (scipy)')
    plt.plot(x_interp, y_custom, '-', label='CubicSpline (custom)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Pomiary i interpolacje (101 punktów)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
