import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Ustawienia siatki
Nx, Ny = 11, 11
dx = dy = 2.0
Lx, Ly = dx * (Nx - 1), dy * (Ny - 1)

# Parametry fizyczne
T_a = 100.0
h_prime = 0.05
coeff = h_prime * dx**2  # 0.2

# Numeracja wewnętrzna (81 węzłów)
def idx(i, j):
    return (j - 1) * (Nx - 2) + (i - 1)

Nint = (Nx - 2) * (Ny - 2)
A = np.zeros((Nint, Nint))
b = np.zeros(Nint)

# Tablica temperatur brzegowych
Tbrzeg = np.full((Ny, Nx), np.nan)
for i in range(Nx):
    Tbrzeg[0, i] = 400 - 10 * i
    Tbrzeg[Ny - 1, i] = 300 - 10 * i
for j in range(Ny):
    Tbrzeg[j, 0] = 400 - 10 * j
    Tbrzeg[j, Nx - 1] = 300 - 10 * j

# Lista kroków (i, j, k)
interior_nodes = [(i, j, idx(i, j)) for j in range(1, Ny - 1) for i in range(1, Nx - 1)]

# Pełna siatka temperatur
T_full = np.copy(Tbrzeg)

# Rozwiązania w trakcie działania
solutions = []

# Rozwiązanie krok po kroku (dla animacji)
for step in range(len(interior_nodes)):
    i, j, k = interior_nodes[step]
    A[k, k] = - (4 + coeff)
    
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if ni == 0 or ni == Nx - 1 or nj == 0 or nj == Ny - 1:
            b[k] -= Tbrzeg[nj, ni]
        else:
            k2 = idx(ni, nj)
            A[k, k2] = 1.0
    
    b[k] += -coeff * T_a
    
    # Rozwiązuj tylko część do aktualnego kroku
    A_sub = A[:k+1, :k+1]
    b_sub = b[:k+1]
    try:
        x_sub = np.linalg.solve(A_sub, b_sub)
    except np.linalg.LinAlgError:
        x_sub = np.zeros(k + 1)
    
    T_temp = np.copy(Tbrzeg)
    for s in range(k + 1):
        si = s % (Nx - 2) + 1
        sj = s // (Nx - 2) + 1
        T_temp[sj, si] = x_sub[s]
    solutions.append(T_temp)

# Utwórz animację
fig, ax = plt.subplots(figsize=(6, 5))
img = ax.imshow(solutions[0], origin='upper', cmap='jet', extent=[0, Lx, Ly, 0], vmin=200, vmax=400)
cbar = fig.colorbar(img, ax=ax, label='Temperatura [°C]')
ax.set_title("Rozkład temperatury – krok po kroku")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

def animate(frame):
    img.set_data(solutions[frame])
    ax.set_title(f"Krok {frame+1}/{len(solutions)} – węzeł (i={interior_nodes[frame][0]}, j={interior_nodes[frame][1]})")

ani = animation.FuncAnimation(fig, animate, frames=len(solutions), interval=300, repeat=False)

# Zapisz animację
final_anim_path = "animacja_rozwiazania.gif"
ani.save(final_anim_path, writer="pillow", fps=4)

final_anim_path
