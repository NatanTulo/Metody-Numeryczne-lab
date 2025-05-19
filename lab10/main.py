import numpy as np
import matplotlib.pyplot as plt

# Definicja równania różniczkowego dy/dt = -(-1 + 4*t**3)*sqrt(y)
def f(t, y):
    if y < 0:
        return 0  # Zapobieganie obliczaniu pierwiastka z liczby ujemnej
    return (1 - 4 * t**3) * np.sqrt(y)

# Rozwiązanie analityczne: y(t) = ((t - t**4 + C)/2)**2, gdzie C = 2*sqrt(2) dla y(0)=2
def y_analytical(t):
    C = 2 * np.sqrt(2)
    return ((t - t**4 + C) / 2) ** 2

# Funkcja znajdująca maksymalną wartość t dla której rozwiązanie istnieje
def find_max_valid_t():
    C = 2 * np.sqrt(2)
    t_test = np.linspace(0, 2, 1000)
    expr = t_test - t_test**4 + C
    valid_indices = np.where(expr >= 0)[0]
    if len(valid_indices) > 0:
        return t_test[valid_indices[-1]]
    return 0

# Parametry
y0 = 2.0         # warunek początkowy
t0 = 0.0         # czas początkowy
n = 100          # liczba segmentów

max_valid_t = find_max_valid_t()
print(f"Maksymalna wartość t dla której rozwiązanie istnieje: {max_valid_t:.4f}")

tk = float(input("Podaj czas końcowy tk: "))
while tk > max_valid_t:
    print(f"Podana wartość tk={tk} jest zbyt duża. Rozwiązanie istnieje tylko dla t <= {max_valid_t:.4f}")
    tk = float(input(f"Podaj czas końcowy tk (nie większy niż {max_valid_t:.4f}): "))

h = (tk - t0) / n

t = np.linspace(t0, tk, n+1)

# Inicjalizacja tablic z rozwiązaniami
y_euler    = np.zeros(n+1)
y_heun     = np.zeros(n+1)
y_midpoint = np.zeros(n+1)

y_euler[0] = y0
y_heun[0] = y0
y_midpoint[0] = y0

# Metoda Eulera
for i in range(n):
    y_next = y_euler[i] + f(t[i], y_euler[i]) * h
    y_euler[i+1] = max(0, y_next) 

# Metoda Heuna (bez iteracji)
for i in range(n):
    k1 = f(t[i], y_heun[i])
    y_pred = y_heun[i] + k1 * h
    y_pred = max(0, y_pred) 
    k2 = f(t[i] + h, y_pred)
    y_next = y_heun[i] + (k1 + k2) / 2 * h
    y_heun[i+1] = max(0, y_next) 

# Metoda punktu środkowego
for i in range(n):
    k1 = f(t[i], y_midpoint[i])
    y_mid = y_midpoint[i] + k1 * (h/2)
    y_mid = max(0, y_mid) 
    k2 = f(t[i] + h/2, y_mid)
    y_next = y_midpoint[i] + k2 * h
    y_midpoint[i+1] = max(0, y_next) 

# Rozwiązanie analityczne
y_exact = y_analytical(t)

# Wykreślanie wyników
plt.figure(figsize=(10, 6))
plt.plot(t, y_exact,     label='Analityczne', linewidth=2)
plt.plot(t, y_euler,     '--', label='Euler')
plt.plot(t, y_heun,      '-.', label='Heun (bez iteracji)')
plt.plot(t, y_midpoint,  ':',  label='Punkt środkowy')

plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Porównanie metod numerycznych i rozwiązania analitycznego')
plt.legend()
plt.grid(True)
plt.show()
