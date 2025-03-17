import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu

# Dane wejściowe
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

# Macierz A
A = np.array([
    [-E_12,   E_12,    0,      0,       0      ],
    [E_12,   -(E_12 + E_23), E_23,  0,       0      ],
    [0,      E_23,  -(E_23 + E_34 + E_35), E_34,  E_35   ],
    [0,      0,      E_34,  -(E_34 + Q_c),  0      ],
    [0,      0,      E_35,  0,     -(E_35 + Q_d) ]
])

# Wektor b
b = np.array([
    -W_s - Q_a * C_a,
    -Q_b * C_b,
    0,
    0,
    -W_g
])

# Rozkład LU macierzy A
P, L, U = lu(A)
print("Macierz L (składnik dolny):")
print(L)
print("Macierz U (składnik górny):")
print(U)

# Rozwiązanie układu Ac = b
c = np.linalg.solve(A, b)

# Wyświetlenie wyników
for i, C in enumerate(c, start=1):
    print(f"C_{i} = {C:.2f} mg/m^3")

# Nowe wartości Ws i Wg
Ws_new = 800   # mg/h
Wg_new = 1200  # mg/h

# Wyznaczenie nowego wektora b z uwzględnieniem ograniczeń
b_new = np.array([
    -Ws_new - Q_a * C_a,
    -Q_b * C_b,
    0,
    0,
    -Wg_new
])

# Rozwiązanie układu Ac = b_new
c_new = np.linalg.solve(A, b_new)

# Wyświetlenie zmienionych wyników
print("\nZmiana stężenia CO przy nowych wartościach:")
for i, C in enumerate(c_new, start=1):
    print(f"C_{i} = {C:.2f} mg/m^3")

# Wyznaczenie macierzy odwrotnej A^-1 metodą LU
I = np.eye(A.shape[0])
A_inv = np.zeros_like(A, dtype=float)
for i in range(A.shape[0]):
    # e to i-ta kolumna macierzy jednostkowej
    e = I[:, i]
    y = np.linalg.solve(L, np.dot(P.T, e))  # P.T = P^-1, bo macierz permutacji
    x = np.linalg.solve(U, y)
    A_inv[:, i] = x
print("\nMacierz odwrotna A^-1:")
print(A_inv)

# Wyznaczanie wkładów poszczególnych źródeł metodą superpozycji
# Przyjmujemy, że:
# - emisja z papierosów (dym) to W_s, pojawia się tylko w pierwszym równaniu,
# - emisja z grilla to W_g, pojawia się tylko w ostatnim równaniu,
# - emisja uliczna to składnik powiązany z przepływami Q_a i Q_b.
b_cig = np.array([
    -W_s,  # emisja z papierosów (dym)
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
    -Q_a * C_a,  # emisja z ulicy (wlot powietrza)
    -Q_b * C_b,
    0,
    0,
    0
])

c_cig = np.linalg.solve(A, b_cig)
c_grill = np.linalg.solve(A, b_grill)
c_street = np.linalg.solve(A, b_street)

# Załóżmy, że pokój dla dzieci odpowiada c4 (indeks 3)
child_conc_cig = c_cig[3]
child_conc_grill = c_grill[3]
child_conc_street = c_street[3]
child_total = child_conc_cig + child_conc_grill + child_conc_street

print("\nUdział procentowy CO w pokoju dla dzieci:")
if child_total != 0:
    pct_cig = 100 * child_conc_cig / child_total
    pct_grill = 100 * child_conc_grill / child_total
    pct_street = 100 * child_conc_street / child_total
    print(f"Papierosy: {pct_cig:.2f}%")
    print(f"Grill: {pct_grill:.2f}%")
    print(f"Ulica: {pct_street:.2f}%")
else:
    print("Brak danych do obliczenia udziałów (suma stężeń wynosi 0).")
