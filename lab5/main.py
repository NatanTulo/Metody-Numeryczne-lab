import numpy as np
import scipy.signal as signal
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt

iloczyn = 1 * 9 * 3 * 5 * 2 * 7
print("Równanie nr: " + str(iloczyn % 24))

# Definicja transmitancji: (z^2 + z + 1) / (z^3 - 3.2*z^2 + 2.75*z - 0.65)
num_coeff = [1, 1, 1]                 # licznik transmitancji
den_coeff = [1, -3.2, 2.75, -0.65]     # mianownik transmitancji

# Obliczanie biegunów transmitancji
poles = np.roots(den_coeff)
print("Bieguny:", poles)

# Sprawdzenie stabilności (układ dyskretny jest stabilny, gdy |pole| < 1)
if np.all(np.abs(poles) < 1):
    print("Układ stabilny")
else:
    print("Układ niestabilny")

# Symulacja odpowiedzi impulsowej dla transmitancji
system_tf = signal.dlti(num_coeff, den_coeff, dt=1)
t, impulse_tf = system_tf.impulse(n=20)
impulse_tf = np.squeeze(impulse_tf)
plt.figure()
plt.stem(t, impulse_tf)
plt.title("Odpowiedź impulsowa - transmitancja")
plt.xlabel("Czas")
plt.ylabel("Amplituda")
plt.grid(True)

#-----------------------------------------------
# Implementacja modelu stanowego w postaci macierzy A, B, C, D
# Na podstawie postaci kanonicznej dla transmitancji:
# G(z) = (c2*z^2 + c1*z + c0) / (z^3 + a2*z^2 + a1*z + a0)
# gdzie:
#   a2 = -3.2, a1 = 2.75, a0 = -0.65
#   c2 = 1, c1 = 1, c0 = 1
a2 = -3.2
a1 = 2.75
a0 = -0.65
A = np.array([
    [-a2, -a1, -a0],  # -a2 = 3.2, -a1 = -2.75, -a0 = 0.65
    [  1,   0,    0],
    [  0,   1,    0]
])
B = np.array([[1],
              [0],
              [0]])
C = np.array([1, 1, 1])  # jako wektor wierszowy
D = 0

print("Macierz A:\n", A)
print("Macierz B:\n", B)
print("Macierz C:\n", C)
print("Macierz D:", D)

# Tworzymy obiekt systemu w postaci modelu stanowego
system_ss = signal.dlti(A, B, C, D, dt=1)

# Wyznaczanie odpowiedzi skokowej dla modelu stanowego (x[0] = 0, u[n] = 1 dla n>=0)
t_step, y_step = signal.dstep(system_ss, n=20)
y_step = np.squeeze(y_step[0])
plt.figure()
plt.stem(t_step, y_step)
plt.title("Odpowiedź skokowa - model stanowy")
plt.xlabel("Czas")
plt.ylabel("Amplituda")
plt.grid(True)

# Implementacja dyskretno-czasowego sterownika LQR
c1 = 1.0  # stała c1
c2 = 1.0  # stała c2
Q = c1 * np.eye(3)      # macierz wagowa Q
R = c2                  # macierz skalara R
P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
print("Wzmocnienie LQR, K:", K)

# Wyznaczanie macierzy P przy użyciu iteracyjnego podstawiania
P_iter = Q.copy()  # inicjalizacja
max_iter = 1000
tol = 1e-8
for i in range(max_iter):
    P_next = A.T @ P_iter @ A - A.T @ P_iter @ B @ np.linalg.inv(R + B.T @ P_iter @ B) @ B.T @ P_iter @ A + Q
    if np.linalg.norm(P_next - P_iter, ord='fro') < tol:
        break
    P_iter = P_next
print("Macierz P iteracyjnie:", P_iter)

# Symulacja odpowiedzi skokowej układu ze sprzężeniem LQR
n_steps = 20
x = np.array([1, 1, 1])  # warunek początkowy
x_history = [x]
for _ in range(n_steps):
    u = -K @ x
    x = A @ x + B[:,0] * u   # B[:,0] jako wektor
    x_history.append(x)
x_history = np.array(x_history)
t_sim = np.arange(n_steps+1)
y_lqr = (C @ x_history.T).flatten()
plt.figure()
plt.stem(t_sim, y_lqr)
plt.title("Odpowiedź skokowa systemu ze sterowaniem LQR")
plt.xlabel("Czas")
plt.ylabel("y")
plt.grid(True)
plt.show()

