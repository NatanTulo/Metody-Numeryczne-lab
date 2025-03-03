import numpy as np
import matplotlib.pyplot as plt

#arcsin(x) = x+x^3/6+x^5/40+...
#przedział x - 0 do 1
#błąd 10^-3

def funSeriesExpansion(n, x):
    m = np.arange(n)
    max_index = 2*(n-1) if n > 0 else 0
    # obliczenie silni od 0! do (2*(n-1))! przy pomocy np.cumprod
    fac = np.concatenate(([1], np.cumprod(np.arange(1, max_index+1)))) if n > 0 else np.array([1])
    coef = fac[2*m] / (4**m * (fac[m]**2) * (2*m + 1))
    return np.sum(coef * x**(2*m+1))

# Dodane: wypisywanie tabeli porównującej wartości rozwinięcia z funkcją arcsin(x)
x = 0.5  # przykładowa wartość x z przedziału [0,1]
true_val = np.arcsin(x)
print(f"{'n':>3} {'Rozwinięcie':>15} {'Błąd bezwzględny':>20} {'Błąd względny [%]':>20}")
for n in range(11):
    approx = funSeriesExpansion(n, x)
    abs_err = abs(approx - true_val)
    rel_err = (abs_err/abs(true_val)*100) if true_val != 0 else 0
    print(f"{n:3d} {approx:15.8f} {abs_err:20.8f} {rel_err:20.8f}")

# Dodane: wykres funkcji & wybranych rozwinięć (n = 0, 2, 7)
x_vals = np.linspace(0, 1, 300)
true_vals = np.arcsin(x_vals)
plt.figure()
plt.plot(x_vals, true_vals, label="arcsin(x)", color="black")
for n_selected in [0, 2, 7]:
    approx_vals = np.array([funSeriesExpansion(n_selected, xv) for xv in x_vals])
    plt.plot(x_vals, approx_vals, label=f"n={n_selected}")
plt.xlabel("x")
plt.ylabel("Wartość funkcji")
plt.legend()
plt.title("Porównanie funkcji arcsin(x) oraz rozwinięć")
plt.show()