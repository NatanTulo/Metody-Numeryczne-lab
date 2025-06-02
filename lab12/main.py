import numpy as np
import matplotlib.pyplot as plt

def main():
    T_a = 100.0                                   
    h_prime = 0.05           
    Lx = 20.0            
    Ly = 20.0            
    Nx = 11                                                
    Ny = 11                                     

    dx = Lx / (Nx - 1)             
    dy = Ly / (Ny - 1)             
    assert abs(dx - dy) < 1e-12, "Δx i Δy muszą być równe"
    delta = dx                         

    coeff = h_prime * delta**2                    

    Nint_x = Nx - 2     
    Nint_y = Ny - 2     
    Nint = Nint_x * Nint_y      


    A = np.zeros((Nint, Nint))
    b = np.zeros(Nint)

    def idx(i, j):
        """
        i, j: indeksy wewnętrzne węzłów (1..9), liczymy od lewej do prawej, od góry ku dołowi.
        Zwraca numer w wektorze x (0..80).
        """
        return (j - 1) * Nint_x + (i - 1)



    Tbrzeg = np.full((Ny, Nx), np.nan)

    for i in range(Nx):
        Tbrzeg[0, i] = 400.0 - 10.0 * i                         

    for j in range(Ny):
        Tbrzeg[j, 0] = 400.0 - 10.0 * j                         

    for i in range(Nx):
        Tbrzeg[Ny - 1, i] = 300.0 - 10.0 * i                    

    for j in range(Ny):
        Tbrzeg[j, Nx - 1] = 300.0 - 10.0 * j                    

    for j in range(1, Ny - 1):                
        for i in range(1, Nx - 1):            
            k = idx(i, j)                             

            A[k, k] = - (4.0 + coeff)          

            if i - 1 == 0:
                b[k] -= Tbrzeg[j, i - 1]
            else:
                k_left = idx(i - 1, j)
                A[k, k_left] = 1.0

            if i + 1 == Nx - 1:
                b[k] -= Tbrzeg[j, i + 1]
            else:
                k_right = idx(i + 1, j)
                A[k, k_right] = 1.0

            if j - 1 == 0:
                b[k] -= Tbrzeg[j - 1, i]
            else:
                k_up = idx(i, j - 1)
                A[k, k_up] = 1.0

            if j + 1 == Ny - 1:
                b[k] -= Tbrzeg[j + 1, i]
            else:
                k_down = idx(i, j + 1)
                A[k, k_down] = 1.0

            b[k] += - coeff * T_a                          

    x_int = np.linalg.solve(A, b)                                                  

    T_full = np.zeros((Ny, Nx))
    T_full[:, :] = Tbrzeg                                                                

    for j in range(1, Ny - 1):              
        for i in range(1, Nx - 1):          
            k = idx(i, j)
            T_full[j, i] = x_int[k]


    plt.figure(figsize=(6, 5))
    img = plt.imshow(
        T_full,
        origin='upper',                   
        cmap='jet',
        extent=[0, Lx, Ly, 0]                                                                   
    )
    plt.colorbar(img, label='Temperatura [°C]')
    plt.title('Rozkład temperatury w stanie ustalonym')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

    xs = np.linspace(0, Lx, Nx)
    ys = np.linspace(0, Ly, Ny)
    for yj in ys:
        for xi in xs:
            plt.plot(xi, yj, 'k.', markersize=3)

    plt.tight_layout()
    plt.show()

    print("Kilka przykładowych temperatur wewnętrznych (wektor x):")
    for idx_prz in [0, 8, 9, 40, 80]:                  
        print(f"  x[{idx_prz:>2d}] = {x_int[idx_prz]:.4f} [°C]")


if __name__ == "__main__":
    main()