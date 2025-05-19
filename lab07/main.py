import numpy as np
import matplotlib.pyplot as plt

try:
    data = np.loadtxt('data7.txt')
except Exception as e:
    print("Błąd wczytywania pliku 'data7.txt':", e)
    exit()

px_meas = data[:, 0]
py_meas = data[:, 1]

T = 1.0

# Macierz przejścia stanu
A = np.array([[1, 0, T, 0],
              [0, 1, 0, T],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Macierz wejścia szumu procesowego
G = np.array([[0, 0],
              [0, 0],
              [1, 0],
              [0, 1]])

# Macierz obserwacji
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Szum procesowy
Q_temp = np.array([[0.25, 0],
                   [0, 0.25]])
Q = G @ Q_temp @ G.T

# Szum pomiarowy
R = np.array([[2, 0],
              [0, 2]])

def run_kalman_filter(px_meas, py_meas, x0, title_suffix=""):
    x_hat = x0.copy()
    P = 5 * np.eye(4)
    
    estimates = [x_hat]
    
    for i in range(len(px_meas) - 1):
        # Faza predykcji (aktualizacja czasu)
        x_hat_pred = A @ x_hat
        P_pred = A @ P @ A.T + Q
        y_hat_pred = H @ x_hat_pred
    
        # Faza aktualizacji (pomiarów)
        y_meas = np.array([px_meas[i+1], py_meas[i+1]])
        innovation = y_meas - y_hat_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_hat = x_hat_pred + K @ innovation
        P = (np.eye(4) - K @ H) @ P_pred
        
        estimates.append(x_hat)
    
    estimates = np.array(estimates)
    
    # Predykcja 5 sekund do przodu
    n_pred = 5
    x_pred_current = x_hat.copy()
    P_pred_future = P.copy()
    
    pred_trajectory = [x_pred_current[:2]]
    
    for _ in range(n_pred):
        x_pred_current = A @ x_pred_current
        P_pred_future = A @ P_pred_future @ A.T + Q
        pred_trajectory.append(x_pred_current[:2])
    
    pred_trajectory = np.array(pred_trajectory)

    px_pred, py_pred = x_pred_current[0], x_pred_current[1]
    print(f"{title_suffix} - Przewidywane położenie po 5s: px = {px_pred:.2f}, py = {py_pred:.2f}")
    
    return estimates, pred_trajectory, (px_pred, py_pred)

# Przypadek 1: Prędkość początkowa zero
x0_zero = np.array([px_meas[0], py_meas[0], 0, 0])
print("Przypadek 1: Prędkość początkowa zero")
estimates_zero, pred_trajectory_zero, final_pred_zero = run_kalman_filter(px_meas, py_meas, x0_zero, "Zerowa prędkość")

# Przypadek 2: Prędkość początkowa obliczona z pierwszych dwóch pomiarów
vx0 = px_meas[1] - px_meas[0]  # bo T = 1.0
vy0 = py_meas[1] - py_meas[0]
x0_calculated = np.array([px_meas[0], py_meas[0], vx0, vy0])
print(f"\nPrzypadek 2: Prędkość początkowa obliczona (vx0={vx0:.2f}, vy0={vy0:.2f})")
estimates_calc, pred_trajectory_calc, final_pred_calc = run_kalman_filter(px_meas, py_meas, x0_calculated, "Obliczona prędkość")

# Wizualizacja porównawcza
plt.figure(figsize=(10, 8))

# Pomiary
plt.plot(px_meas, py_meas, 'rx', label='Pomiary')

# Przypadek 1: Zerowa prędkość początkowa
plt.plot(estimates_zero[:, 0], estimates_zero[:, 1], 'b-', label='Estymowana (v0=0)', linewidth=2)
plt.plot(pred_trajectory_zero[:, 0], pred_trajectory_zero[:, 1], 'b--', linewidth=1.5)
plt.plot(final_pred_zero[0], final_pred_zero[1], 'bo', markersize=8)

# Przypadek 2: Obliczona prędkość początkowa
plt.plot(estimates_calc[:, 0], estimates_calc[:, 1], 'g-', label=f'Estymowana (v[0] = (p[1]-p[0])/T)', linewidth=2)
plt.plot(pred_trajectory_calc[:, 0], pred_trajectory_calc[:, 1], 'g--', linewidth=1.5)
plt.plot(final_pred_calc[0], final_pred_calc[1], 'go', markersize=8)

plt.xlabel('px')
plt.ylabel('py')
plt.title('Porównanie trajektorii dla różnych prędkości początkowych')
plt.legend()
plt.grid()
plt.show()

# Różnica w predykcji po 5s
px_diff = final_pred_calc[0] - final_pred_zero[0]
py_diff = final_pred_calc[1] - final_pred_zero[1]
print(f"\nRóżnica w predykcji po 5s: Δpx = {px_diff:.2f}, Δpy = {py_diff:.2f}")