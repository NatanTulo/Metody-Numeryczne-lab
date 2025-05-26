import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Parametry zadania
t0 = 0          # czas początkowy
tk = 25         # czas końcowy
h = 0.03125     # krok metody
x0, y0, z0 = 5, 5, 5  # warunki początkowe

# Parametry układu Lorenza (klasyczny układ równań różniczkowych)
sigma = 10.0
rho = 28.0
beta = 8.0/3.0

def lorenz_system(t, state):
    """
    Układ równań Lorenza:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# 1. Rozwiązanie za pomocą gotowej metody z scipy (1 pkt)
print("=== Rozwiązywanie za pomocą scipy.integrate.solve_ivp ===")

# Wektor czasu
t_span = (t0, tk)
t_eval = np.arange(t0, tk + h, h)
initial_conditions = [x0, y0, z0]

# Rozwiązanie z użyciem metody RK45 (podobnej do RK4)
sol_scipy = solve_ivp(lorenz_system, t_span, initial_conditions, 
                     t_eval=t_eval, method='RK45', rtol=1e-8)

t_scipy = sol_scipy.t
x_scipy, y_scipy, z_scipy = sol_scipy.y

print(f"Rozwiązanie scipy - liczba punktów: {len(t_scipy)}")
print(f"Ostatnie wartości: x={x_scipy[-1]:.4f}, y={y_scipy[-1]:.4f}, z={z_scipy[-1]:.4f}")

# 2. Własna implementacja metody Rungego-Kutty 4. rzędu (3 pkt)
print("\n=== Własna implementacja metody Rungego-Kutty 4. rzędu ===")

def runge_kutta_4(f, t0, tk, y0, h):
    """
    Metoda Rungego-Kutty 4. rzędu dla układu równań różniczkowych
    
    Parametry:
    f - funkcja prawej strony układu równań dy/dt = f(t, y)
    t0 - czas początkowy
    tk - czas końcowy
    y0 - warunek początkowy (wektor)
    h - krok całkowania
    
    Zwraca:
    t - wektor czasów
    y - macierz rozwiązań (każdy wiersz to jedna zmienna)
    """
    
    # Liczba kroków
    n_steps = int((tk - t0) / h) + 1
    t = np.linspace(t0, tk, n_steps)
    
    # Inicjalizacja
    y = np.zeros((len(y0), n_steps))
    y[:, 0] = y0
    
    # Główna pętla metody RK4
    for i in range(n_steps - 1):
        # Obecny stan
        t_i = t[i]
        y_i = y[:, i]
        
        # Współczynniki k1, k2, k3, k4
        k1 = h * f(t_i, y_i)
        k2 = h * f(t_i + h/2, y_i + k1/2)
        k3 = h * f(t_i + h/2, y_i + k2/2)
        k4 = h * f(t_i + h, y_i + k3)
        
        # Nowy stan zgodnie ze wzorem RK4
        y[:, i+1] = y_i + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y

# Rozwiązanie własną metodą RK4
t_rk4, y_rk4 = runge_kutta_4(lorenz_system, t0, tk, initial_conditions, h)
x_rk4, y_rk4_var, z_rk4 = y_rk4

print(f"Własna metoda RK4 - liczba punktów: {len(t_rk4)}")
print(f"Ostatnie wartości: x={x_rk4[-1]:.4f}, y={y_rk4_var[-1]:.4f}, z={z_rk4[-1]:.4f}")

# 3. Przedstawienie przebiegu zmiennych x, y, z na trzech osobnych wykresach (1 pkt)
print("\n=== Tworzenie wykresów przebiegu zmiennych ===")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Wykres x(t)
ax1.plot(t_scipy, x_scipy, 'b-', label='scipy RK45', linewidth=1.5)
ax1.plot(t_rk4, x_rk4, 'r--', label='własna RK4', linewidth=1.5, alpha=0.8)
ax1.set_xlabel('Czas t')
ax1.set_ylabel('x(t)')
ax1.set_title('Przebieg zmiennej x(t)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Wykres y(t)
ax2.plot(t_scipy, y_scipy, 'b-', label='scipy RK45', linewidth=1.5)
ax2.plot(t_rk4, y_rk4_var, 'r--', label='własna RK4', linewidth=1.5, alpha=0.8)
ax2.set_xlabel('Czas t')
ax2.set_ylabel('y(t)')
ax2.set_title('Przebieg zmiennej y(t)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Wykres z(t)
ax3.plot(t_scipy, z_scipy, 'b-', label='scipy RK45', linewidth=1.5)
ax3.plot(t_rk4, z_rk4, 'r--', label='własna RK4', linewidth=1.5, alpha=0.8)
ax3.set_xlabel('Czas t')
ax3.set_ylabel('z(t)')
ax3.set_title('Przebieg zmiennej z(t)')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()

# 4. Trajektoria fazowa w przestrzeni trójwymiarowej (1 pkt)
print("\n=== Tworzenie trajektorii fazowej 3D ===")

fig = plt.figure(figsize=(14, 6))

# Trajektoria scipy
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x_scipy, y_scipy, z_scipy, 'b-', linewidth=1.5, label='scipy RK45')
ax1.scatter([x0], [y0], [z0], color='green', s=100, label='Start')
ax1.scatter([x_scipy[-1]], [y_scipy[-1]], [z_scipy[-1]], color='red', s=100, label='Koniec')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Trajektoria fazowa 3D - scipy RK45')
ax1.legend()

# Trajektoria własna RK4
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(x_rk4, y_rk4_var, z_rk4, 'r-', linewidth=1.5, label='własna RK4')
ax2.scatter([x0], [y0], [z0], color='green', s=100, label='Start')
ax2.scatter([x_rk4[-1]], [y_rk4_var[-1]], [z_rk4[-1]], color='red', s=100, label='Koniec')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('Trajektoria fazowa 3D - własna RK4')
ax2.legend()

plt.tight_layout()
plt.show()

# Porównanie obu metod
print("\n=== Porównanie metod ===")
print(f"Maksymalna różnica w x: {np.max(np.abs(x_scipy - x_rk4)):.6f}")
print(f"Maksymalna różnica w y: {np.max(np.abs(y_scipy - y_rk4_var)):.6f}")
print(f"Maksymalna różnica w z: {np.max(np.abs(z_scipy - z_rk4)):.6f}")

# Analiza właściwości rozwiązania
print(f"\n=== Analiza rozwiązania ===")
print(f"Zakres wartości x: [{np.min(x_rk4):.3f}, {np.max(x_rk4):.3f}]")
print(f"Zakres wartości y: [{np.min(y_rk4_var):.3f}, {np.max(y_rk4_var):.3f}]")
print(f"Zakres wartości z: [{np.min(z_rk4):.3f}, {np.max(z_rk4):.3f}]")

# Dodatkowy wykres porównawczy trajektorii
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Obie trajektorie na jednym wykresie
ax.plot(x_scipy, y_scipy, z_scipy, 'b-', linewidth=2, label='scipy RK45', alpha=0.8)
ax.plot(x_rk4, y_rk4_var, z_rk4, 'r--', linewidth=2, label='własna RK4', alpha=0.8)
ax.scatter([x0], [y0], [z0], color='green', s=150, label='Start', zorder=5)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('z', fontsize=12)
ax.set_title('Porównanie trajektorii fazowych 3D\nUkład Lorenza', fontsize=14)
ax.legend(fontsize=12)

# Ulepszenie wyglądu wykresu 3D
ax.grid(True, alpha=0.3)
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

print(f"\n=== Podsumowanie ===")
print(f"Zadanie wykonane pomyślnie!")
print(f"- Użyto metody scipy.integrate.solve_ivp (RK45)")
print(f"- Zaimplementowano własną metodę Rungego-Kutty 4. rzędu")
print(f"- Przedstawiono przebiegi zmiennych x(t), y(t), z(t)")
print(f"- Wykreślono trajektorie fazowe w przestrzeni 3D")
print(f"- Parametry: t0={t0}, tk={tk}, h={h}, warunki początkowe=({x0},{y0},{z0})")