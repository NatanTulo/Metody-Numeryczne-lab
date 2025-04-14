import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Wczytanie danych z pliku
# -------------------------
# Zakładamy, że plik 'data7.txt' zawiera pomiary (px, py) oddzielone spacjami lub średnikami.
# W razie potrzeby można zmodyfikować separator w funkcji np.loadtxt (argument delimiter)
try:
    # Jeśli dane są oddzielone spacjami
    data = np.loadtxt('data7.txt')
except Exception as e:
    print("Błąd wczytywania pliku 'data7.txt':", e)
    exit()

# Załóżmy, że dane mają dwie kolumny: [px, py]
px_meas = data[:, 0]
py_meas = data[:, 1]

# -------------------------
# Parametry modelu
# -------------------------
T = 1.0  # Okres próbkowania, 1 sekunda

# Macierz przejścia stanu (A) dla modelu 2D:
# x = [px, py, vx, vy]^T
A = np.array([[1, 0, T, 0],
              [0, 1, 0, T],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Macierz wejścia szumu procesowego (G)
G = np.array([[0, 0],
              [0, 0],
              [1, 0],
              [0, 1]])

# Macierz obserwacji (H) – czujnik mierzy tylko położenie (px, py)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# -------------------------
# Parametry szumów
# -------------------------
# Szum procesowy: q ~ N(0, Q_temp) – zadany jako:
Q_temp = np.array([[0.25, 0],
                   [0, 0.25]])
# Po przekształceniu przez macierz G: Q = G·Q_temp·Gᵀ
Q = G @ Q_temp @ G.T

# Szum pomiarowy: w ~ N(0, R)
R = np.array([[2, 0],
              [0, 2]])

# -------------------------
# Inicjalizacja stanu i macierzy kowariancji
# -------------------------
# Stan początkowy na podstawie pierwszego pomiaru, przyjmując prędkość początkową v[0] = 0
x0 = np.array([px_meas[0], py_meas[0], 0, 0])
x_hat = x0  # Estymata stanu
# Macierz kowariancji początkowej: P[0|0] = 5·I
P = 5 * np.eye(4)

# Zapisywanie wyników
estimates = [x_hat]

# -------------------------
# Implementacja filtru Kalmana
# -------------------------
# Iterujemy dla kolejnych pomiarów (pomiar wykonany co sekundę)
for i in range(len(px_meas) - 1):
    # ---------- Faza predykcji (aktualizacja czasu) ----------
    # 1. Predykcja stanu: x̂[n+1|n] = A * x̂[n|n]
    x_hat_pred = A @ x_hat
    
    # 2. Predykcja macierzy kowariancji: P[n+1|n] = A * P[n|n] * Aᵀ + G·Q·Gᵀ
    P_pred = A @ P @ A.T + Q
    
    # 3. Predykcja pomiaru: ŷ[n+1|n] = H * x̂[n+1|n]
    y_hat_pred = H @ x_hat_pred
    
    # ---------- Faza aktualizacji (pomiarów) ----------
    # Odczyt pomiaru w chwili n+1
    y_meas = np.array([px_meas[i+1], py_meas[i+1]])
    
    # 4. Obliczenie innowacji: e[n+1] = y[n+1] − ŷ[n+1|n]
    innovation = y_meas - y_hat_pred
    
    # 5. Obliczenie macierzy kowariancji innowacji: S[n+1] = H·P[n+1|n]·Hᵀ + R
    S = H @ P_pred @ H.T + R
    
    # 6. Obliczenie wzmocnienia Kalmana: K[n+1] = P[n+1|n]·Hᵀ·(S[n+1])⁻¹
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # 7. Aktualizacja estymaty stanu: x̂[n+1|n+1] = x̂[n+1|n] + K[n+1]*e[n+1]
    x_hat = x_hat_pred + K @ innovation
    
    # 8. Aktualizacja macierzy kowariancji: P[n+1|n+1] = (I - K[n+1]*H)·P[n+1|n]
    P = (np.eye(4) - K @ H) @ P_pred
    
    estimates.append(x_hat)

# Przekształcamy listę estymat na macierz numpy dla wygodniejszego użycia przy wykresie
estimates = np.array(estimates)

# -------------------------
# Wizualizacja wyników
# -------------------------
plt.figure(figsize=(8, 6))
# Rzeczywiste pomiary oznaczamy jako czerwone kropki
plt.plot(px_meas, py_meas, 'r.', label='Pomiary')
# Estymowaną trajektorię oznaczamy jako niebieską linią
plt.plot(estimates[:, 0], estimates[:, 1], 'b-', label='Estymowana trajektoria')
plt.xlabel('px')
plt.ylabel('py')
plt.title('Trajektoria samolotu - filtr Kalmana')
plt.legend()
plt.grid()
plt.show()
