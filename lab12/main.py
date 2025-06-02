import numpy as np
import matplotlib.pyplot as plt

def main():
    # Parametry fizyczne
    T_a = 100.0     # [°C] – temperatura otoczenia
    h_prime = 0.05  # [1/m^2]
    Lx = 20.0       # [m]
    Ly = 20.0       # [m]
    Nx = 11         # liczba węzłów w kierunku x (0,2,…,20)
    Ny = 11         # liczba węzłów w kierunku y
    
    # Krok siatki
    dx = Lx / (Nx - 1)  # = 2.0 [m]
    dy = Ly / (Ny - 1)  # = 2.0 [m]
    assert abs(dx - dy) < 1e-12, "Δx i Δy muszą być równe"
    delta = dx  # wspólna wartość kroku
    
    # h' * (Δx)^2
    coeff = h_prime * delta**2  # 0.05 * 2^2 = 0.2
    
    # Liczba wewnętrznych węzłów (9 × 9 = 81)
    Nint_x = Nx - 2  # 9
    Nint_y = Ny - 2  # 9
    Nint = Nint_x * Nint_y  # 81
    
    # --- 1) Budujemy macierz A (81×81) i wektor b (81×1) ---
    
    A = np.zeros((Nint, Nint))
    b = np.zeros(Nint)
    
    # Funkcja pomocnicza: z indeksu (i,j) (i,j = 1..9) na numer w wektorze 0..80
    def idx(i, j):
        """
        i, j: indeksy wewnętrzne węzłów (1..9), liczymy od lewej do prawej, od góry ku dołowi.
        Zwraca numer w wektorze x (0..80).
        """
        return (j - 1) * Nint_x + (i - 1)
    
    # Najpierw definiujemy macierz A i wektor b wypełnione zerami,
    # potem wypełniamy je równaniami dla każdego wnętrznego węzła (i,j).
    
    # Znamy wartości brzegowe w 11×11 – przygotujmy tablicę Tbrzeg, by łatwo
    # uzyskiwać temperaturę w węźle brzegowym o współrzędnej (i,j) = (0..10,0..10).
    # Inicjalizujemy całą tablicę 11×11 jako None, a potem nadpisujemy brzeg.
    
    Tbrzeg = np.full((Ny, Nx), np.nan)
    # Wypełniamy brzegi:
    
    # Górna krawędź (y=0, j=0), x = 0..10
    # wartości: [400, 390, 380, …, 300]
    for i in range(Nx):
        Tbrzeg[0, i] = 400.0 - 10.0 * i  # 400, 390, 380, …, 300
    
    # Lewy bok (x=0, i=0), y=0..10
    # wartości: [400, 390, 380, …, 300]
    for j in range(Ny):
        Tbrzeg[j, 0] = 400.0 - 10.0 * j  # 400, 390, 380, …, 300
    
    # Dolna krawędź (y=10, j=10), x=0..10
    # wartości: [300, 290, 280, …, 200]
    for i in range(Nx):
        Tbrzeg[Ny - 1, i] = 300.0 - 10.0 * i  # 300, 290, …, 200
    
    # Prawy bok (x=10, i=10), y=0..10
    # wartości: [300, 290, 280, …, 200]
    for j in range(Ny):
        Tbrzeg[j, Nx - 1] = 300.0 - 10.0 * j  # 300, 290, …, 200
    
    # Teraz przechodzimy po wszystkich wnętrznych (i,j), gdzie i,j = 1..9
    for j in range(1, Ny - 1):      # j = 1..9
        for i in range(1, Nx - 1):  # i = 1..9
            k = idx(i, j)  # numer równania od 0 do 80
            
            # Współczynnik centralny:
            A[k, k] = - (4.0 + coeff)  # = -4.2
            
            # 1) sąsiad od lewej: (i-1, j)
            if i - 1 == 0:
                # to jest węzeł brzegowy – wartość w Tbrzeg[j,i-1]
                b[k] -= Tbrzeg[j, i - 1]
            else:
                # wewnętrzny
                k_left = idx(i - 1, j)
                A[k, k_left] = 1.0
            
            # 2) sąsiad od prawej: (i+1, j)
            if i + 1 == Nx - 1:
                # węzeł brzegowy – wartość w Tbrzeg[j, i+1]
                b[k] -= Tbrzeg[j, i + 1]
            else:
                k_right = idx(i + 1, j)
                A[k, k_right] = 1.0
            
            # 3) sąsiad z góry: (i, j-1)
            if j - 1 == 0:
                # węzeł brzegowy
                b[k] -= Tbrzeg[j - 1, i]
            else:
                k_up = idx(i, j - 1)
                A[k, k_up] = 1.0
            
            # 4) sąsiad z dołu: (i, j+1)
            if j + 1 == Ny - 1:
                # węzeł brzegowy
                b[k] -= Tbrzeg[j + 1, i]
            else:
                k_down = idx(i, j + 1)
                A[k, k_down] = 1.0
            
            # 5) dodajemy do b stałą (→ prawa strona: −20)
            b[k] += - coeff * T_a  # czyli -0.2 * 100 = -20
    
    # --- 2) Rozwiązujemy układ A x = b ---
    # Używamy np.linalg.solve (macierz jest 81×81, niesinglularna w naszym przypadku)
    x_int = np.linalg.solve(A, b)  # wektor długości 81 – to temperatury wewnętrzne
    
    # Teraz złożymy pełną macierz temperatur 11×11:
    T_full = np.zeros((Ny, Nx))
    # Najpierw skopiujmy brzegi (zdefiniowane w Tbrzeg)
    T_full[:, :] = Tbrzeg  # tu uwaga: Tbrzeg ma już wypełnione tylko brzegi, wnętrze NaN
    
    # Przepisujemy wnętrze:
    for j in range(1, Ny - 1):      # j=1..9
        for i in range(1, Nx - 1):  # i=1..9
            k = idx(i, j)
            T_full[j, i] = x_int[k]
    
    # --- 3) Rysujemy kolorowy rozkład temperatury ---
    # Zrobimy obrazek za pomocą plt.imshow(), pamiętając że pierwszy indeks to y,
    # a drugi indeks to x. Domyślnie w imshow pionowa oś idzie od góry do dołu,
    # ale chcemy, żeby y=0 u góry (zgodnie z rysunkiem – y rośnie w dół). To się domyślnie zgadza:
    
    plt.figure(figsize=(6, 5))
    # Użyjemy cmap='jet' (lub inną), interpolation='nearest'
    img = plt.imshow(
        T_full,
        origin='upper',  # y=0 na wierzchu
        cmap='jet',
        extent=[0, Lx, Ly, 0]  # [xmin, xmax, ymin, ymax] - odwrócone ymax→ymin, żeby y=0 u góry
    )
    plt.colorbar(img, label='Temperatura [°C]')
    plt.title('Rozkład temperatury w stanie ustalonym')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    
    # Dokładnie na węzłach naniesiemy czarne kropki
    # Węzły: x_i = i*2, y_j = j*2
    xs = np.linspace(0, Lx, Nx)
    ys = np.linspace(0, Ly, Ny)
    # dla każdego y, x nakładamy kropkę
    for yj in ys:
        for xi in xs:
            plt.plot(xi, yj, 'k.', markersize=3)
    
    plt.tight_layout()
    plt.show()
    
    # (opcjonalnie) Drukujemy kilka przykładów uzyskanych temperatur wewnętrznych
    print("Kilka przykładowych temperatur wewnętrznych (wektor x):")
    for idx_prz in [0, 8, 9, 40, 80]:  # krotki podgląd
        print(f"  x[{idx_prz:>2d}] = {x_int[idx_prz]:.4f} [°C]")

    
if __name__ == "__main__":
    main()
