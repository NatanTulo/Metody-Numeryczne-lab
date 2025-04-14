import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Wczytanie danych z pliku
# -------------------------
try:
    data = np.loadtxt('data7.txt')
except Exception as e:
    print("Błąd wczytywania pliku 'data7.txt':", e)
    exit()

px_meas = data[:, 0]
py_meas = data[:, 1]

# -------------------------
# Parametry modelu
# -------------------------
T = 1.0  # okres próbkowania = 1 sekunda

# Macierz przejścia stanu dla modelu 2D (stan: [px, py, vx, vy]^T)
A = np.array([[1, 0, T, 0],
              [0, 1, 0, T],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Macierz wejścia szumu procesowego
G = np.array([[0, 0],
              [0, 0],
              [1, 0],
              [0, 1]])

# Macierz obserwacji – pomiar tylko położenia (px, py)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# -------------------------
# Parametry szumów
# -------------------------
# Szum procesowy podany jako Q_temp, następnie transformowany przez G
Q_temp = np.array([[0.25, 0],
                   [0, 0.25]])
Q = G @ Q_temp @ G.T

# Szum pomiarowy
R = np.array([[2, 0],
              [0, 2]])

# -------------------------
# Funkcja implementująca filtr Kalmana
# -------------------------
def run_kalman_filter(px_meas, py_meas, x0, title_suffix=""):
    x_hat = x0.copy()
    P = 5 * np.eye(4)  # P[0|0] = 5·I
    
    # Tablica do zapisywania estymat
    estimates = [x_hat]
    
    # Pętla filtru Kalmana
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
    
    # Przekształcenie listy estymat do macierzy numpy
    estimates = np.array(estimates)
    
    # -------------------------
    # Metoda 1: Bezpośrednia predykcja na 5 sekund w przód
    # -------------------------
    n_pred = 5
    x_pred_current = x_hat.copy()
    P_pred_future = P.copy()
    
    # Lista do przechowywania punktów przewidywanej trajektorii
    pred_trajectory_direct = [x_pred_current[:2]]
    
    # Dokonujemy predykcji przez kolejne 5 kroków
    for _ in range(n_pred):
        x_pred_current = A @ x_pred_current
        P_pred_future = A @ P_pred_future @ A.T + Q
        pred_trajectory_direct.append(x_pred_current[:2])
    
    pred_trajectory_direct = np.array(pred_trajectory_direct)
    px_pred_direct, py_pred_direct = x_pred_current[0], x_pred_current[1]
    print(f"{title_suffix} - Metoda 1 - Przewidywane położenie po 5s: px = {px_pred_direct:.2f}, py = {py_pred_direct:.2f}")
    
    # -------------------------
    # Metoda 2: Sekwencyjna predykcja co 1 sekundę z aktualizacją
    # -------------------------
    x_pred_seq = x_hat.copy()
    P_pred_seq = P.copy()
    
    # Lista do przechowywania punktów przewidywanej trajektorii dla metody sekwencyjnej
    pred_trajectory_seq = [x_pred_seq[:2]]
    
    for i in range(n_pred):
        # Faza predykcji na 1s
        x_pred_seq = A @ x_pred_seq
        P_pred_seq = A @ P_pred_seq @ A.T + Q
        
        # Generujemy sztuczny "pomiar" na podstawie predykcji
        # Dodajemy losowe zakłócenie, aby symulować rzeczywiste warunki
        noise = np.random.multivariate_normal([0, 0], R)
        fake_measurement = H @ x_pred_seq + noise
        
        # Faza aktualizacji - traktujemy predykcję z szumem jako nowy pomiar
        y_meas = fake_measurement
        y_pred = H @ x_pred_seq
        innovation = y_meas - y_pred
        S = H @ P_pred_seq @ H.T + R
        K = P_pred_seq @ H.T @ np.linalg.inv(S)
        x_pred_seq = x_pred_seq + K @ innovation
        P_pred_seq = (np.eye(4) - K @ H) @ P_pred_seq
        
        pred_trajectory_seq.append(x_pred_seq[:2])
    
    pred_trajectory_seq = np.array(pred_trajectory_seq)
    px_pred_seq, py_pred_seq = x_pred_seq[0], x_pred_seq[1]
    print(f"{title_suffix} - Metoda 2 - Przewidywane położenie po 5s: px = {px_pred_seq:.2f}, py = {py_pred_seq:.2f}")
    
    return estimates, (pred_trajectory_direct, (px_pred_direct, py_pred_direct)), (pred_trajectory_seq, (px_pred_seq, py_pred_seq))

# -------------------------
# Przypadek 1: Prędkość początkowa zero
# -------------------------
x0_zero = np.array([px_meas[0], py_meas[0], 0, 0])
print("Przypadek 1: Prędkość początkowa zero")
estimates_zero, direct_pred_zero, seq_pred_zero = run_kalman_filter(px_meas, py_meas, x0_zero, "Zerowa prędkość")

# -------------------------
# Przypadek 2: Prędkość początkowa obliczona z pierwszych dwóch pomiarów
# -------------------------
vx0 = px_meas[1] - px_meas[0]  # bo T = 1.0
vy0 = py_meas[1] - py_meas[0]
x0_calculated = np.array([px_meas[0], py_meas[0], vx0, vy0])
print(f"\nPrzypadek 2: Prędkość początkowa obliczona (vx0={vx0:.2f}, vy0={vy0:.2f})")
estimates_calc, direct_pred_calc, seq_pred_calc = run_kalman_filter(px_meas, py_meas, x0_calculated, "Obliczona prędkość")

# -------------------------
# Wizualizacja porównawcza
# -------------------------
plt.figure(figsize=(12, 9))

# Pomiary
plt.plot(px_meas, py_meas, 'rx', label='Pomiary')

# Przypadek 1: Zerowa prędkość początkowa
plt.plot(estimates_zero[:, 0], estimates_zero[:, 1], 'b-', label='Estymowana (v0=0)', linewidth=2)
# Metoda 1 (bezpośrednia)
plt.plot(direct_pred_zero[0][:, 0], direct_pred_zero[0][:, 1], 'b--', linewidth=1.5, label='Predykcja bezpośrednia (v0=0)')
plt.plot(direct_pred_zero[1][0], direct_pred_zero[1][1], 'bo', markersize=10)
# Metoda 2 (sekwencyjna)
plt.plot(seq_pred_zero[0][:, 0], seq_pred_zero[0][:, 1], 'b:', linewidth=1.5, label='Predykcja sekwencyjna (v0=0)')
plt.plot(seq_pred_zero[1][0], seq_pred_zero[1][1], 'b*', markersize=10)

# Przypadek 2: Obliczona prędkość początkowa
plt.plot(estimates_calc[:, 0], estimates_calc[:, 1], 'g-', label=f'Estymowana (v[0] = (p[1]-p[0])/T)', linewidth=2)
# Metoda 1 (bezpośrednia)
plt.plot(direct_pred_calc[0][:, 0], direct_pred_calc[0][:, 1], 'g--', linewidth=1.5, label='Predykcja bezpośrednia (v0=obliczona)')
plt.plot(direct_pred_calc[1][0], direct_pred_calc[1][1], 'go', markersize=10)
# Metoda 2 (sekwencyjna)
plt.plot(seq_pred_calc[0][:, 0], seq_pred_calc[0][:, 1], 'g:', linewidth=1.5, label='Predykcja sekwencyjna (v0=obliczona)')
plt.plot(seq_pred_calc[1][0], seq_pred_calc[1][1], 'g*', markersize=10)

plt.xlabel('px')
plt.ylabel('py')
plt.title('Porównanie metod predykcji dla różnych prędkości początkowych')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Wyświetl różnice w predykcjach
print("\nRóżnice w predykcjach po 5s między metodami:")
print(f"v0=0 - Różnica: Δpx = {seq_pred_zero[1][0] - direct_pred_zero[1][0]:.2f}, Δpy = {seq_pred_zero[1][1] - direct_pred_zero[1][1]:.2f}")
print(f"v0=obliczona - Różnica: Δpx = {seq_pred_calc[1][0] - direct_pred_calc[1][0]:.2f}, Δpy = {seq_pred_calc[1][1] - direct_pred_calc[1][1]:.2f}")
