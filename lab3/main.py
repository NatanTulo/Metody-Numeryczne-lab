import numpy as np

# 1. Dane wejściowe oraz parametry (zgodnie z przykładem z wykładu)
Q_a = 200   # m^3/h
Q_b = 300   # m^3/h
Q_c = 150   # m^3/h
Q_d = 350   # m^3/h

W_s = 1500  # mg/h
W_g = 2500  # mg/h
C_a = 2     # mg/m^3
C_b = 2     # mg/m^3

E_12 = 25   # m^3/h
E_23 = 50   # m^3/h
E_34 = 50   # m^3/h
E_35 = 25   # m^3/h

# 2. Definicja macierzy A i wektora b (układ równań)
A = np.array([
    [-E_12,         E_12,              0,            0,              0],
    [E_12, -(E_12 + E_23),          E_23,            0,              0],
    [0,              E_23, -(E_23 + E_34 + E_35),   E_34,        E_35],
    [0,               0,              E_34, -(E_34 + Q_c),       0],
    [0,               0,              E_35,            0,   -(E_35 + Q_d)]
])
b = np.array([
    -W_s - Q_a * C_a,
    -Q_b * C_b,
    0,
    0,
    -W_g
])

def lu_decomposition(A):
    """LU decomposition with partial pivoting.
    Returns permutation matrix P, lower triangular matrix L (unit diagonal) and upper triangular matrix U such that PA = LU.
    """
    A = A.copy()
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()
    
    for i in range(n):
        pivot = np.argmax(abs(U[i:, i])) + i
        if U[pivot, i] == 0:
            raise ValueError("Matrix is singular.")
        if pivot != i:
            U[[i, pivot], :] = U[[pivot, i], :]
            P[[i, pivot], :] = P[[pivot, i], :]
            if i >= 1:
                L[[i, pivot], :i] = L[[pivot, i], :i]
        L[i, i] = 1.0
        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, :] = U[j, :] - L[j, i] * U[i, :]
    return P, L, U

# 3. Rozkład LU macierzy A bez korzystania z scipy
P, L, U = lu_decomposition(A)
print("Macierz L (dolny składnik):")
print(L)
print("Macierz U (górny składnik):")
print(U)

# 4. Rozwiązanie układu Ac = b metodą podstawienia (najpierw rozwiązujemy L*y = P^T * b, potem U*c = y)
y = np.linalg.solve(L, np.dot(P.T, b))
c = np.linalg.solve(U, y)
print("\nRozwiązanie układu równań (stężenia CO):")
for i, Ci in enumerate(c, start=1):
    print(f"C_{i} = {Ci:.2f} mg/m^3")

# 5. Wyznaczenie macierzy odwrotnej A^-1 metodą LU (kolumna po kolumnie)
I = np.eye(A.shape[0])
A_inv = np.zeros_like(A, dtype=float)
for i in range(A.shape[0]):
    e = I[:, i]
    y_e = np.linalg.solve(L, np.dot(P.T, e))
    x_e = np.linalg.solve(U, y_e)
    A_inv[:, i] = x_e
print("\nMacierz odwrotna A^-1:")
print(A_inv)

# 6. Rozwiązanie układu dla zmienionych warunków emisji
Ws_new = 800   # mg/h
Wg_new = 1200  # mg/h
b_new = np.array([
    -Ws_new - Q_a * C_a,
    -Q_b * C_b,
    0,
    0,
    -Wg_new
])
y_new = np.linalg.solve(L, np.dot(P.T, b_new))
c_new = np.linalg.solve(U, y_new)
print("\nRozwiązanie układu dla nowych wartości emisji:")
for i, Ci in enumerate(c_new, start=1):
    print(f"C_{i} = {Ci:.2f} mg/m^3")

# 7. Wyznaczenie wkładów poszczególnych źródeł metodą superpozycji
# Definiujemy oddzielne wektory prawych stron dla:
# - emisji z papierosów (dym): pojawia się tylko w pierwszym równaniu,
# - emisji z grilla: pojawia się tylko w ostatnim równaniu,
# - wpływu ulicznego: związany z przepływami Q_a i Q_b.
b_cig = np.array([
    -W_s,  # emisja papierosowa
    0,
    0,
    0,
    0
])
b_grill = np.array([
    0,
    0,
    0,
    0,
    -W_g  # emisja z grilla
])
b_street = np.array([
    -Q_a * C_a,  # wpływ uliczny
    -Q_b * C_b,
    0,
    0,
    0
])

y_cig = np.linalg.solve(L, np.dot(P.T, b_cig))
c_cig = np.linalg.solve(U, y_cig)
y_grill = np.linalg.solve(L, np.dot(P.T, b_grill))
c_grill = np.linalg.solve(U, y_grill)
y_street = np.linalg.solve(L, np.dot(P.T, b_street))
c_street = np.linalg.solve(U, y_street)

child_conc_cig = c_cig[3]
child_conc_grill = c_grill[3]
child_conc_street = c_street[3]
child_total = child_conc_cig + child_conc_grill + child_conc_street

print("\nUdział procentowy poszczególnych źródeł w pokoju dla dzieci:")
if child_total != 0:
    pct_cig = 100 * child_conc_cig / child_total
    pct_grill = 100 * child_conc_grill / child_total
    pct_street = 100 * child_conc_street / child_total
    print(f"Papierosy: {pct_cig:.2f}%")
    print(f"Grill: {pct_grill:.2f}%")
    print(f"Ulica: {pct_street:.2f}%")
else:
    print("Brak danych do obliczenia udziałów (suma stężeń wynosi 0).")
