import numpy as np
import matplotlib.pyplot as plt
import io

#błąd 10^-3

def funSeriesExpansion(n, x):
    x = np.array(x)  # konwersja do numpy array
    m = np.arange(n)
    max_index = 2*(n-1) if n > 0 else 0
    # obliczenie silni od 0! do (2*(n-1))! przy pomocy np.cumprod
    fac = np.concatenate(([1], np.cumprod(np.arange(1, max_index+1)))) if n > 0 else np.array([1])
    coef = fac[2*m] / (4**m * (fac[m]**2) * (2*m + 1))
    if x.ndim == 0:  # x jest skalarem
        return np.sum(coef * x**(2*m+1))
    else:
        return np.sum(coef * np.power(x.reshape(-1, 1), (2*m+1).reshape(1, -1)), axis=1)

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

def testSeriesRandom():
    # Generowanie 5 losowych punktów x z przedziału [0,1] (bez pętli)
    x_tests = np.random.uniform(0, 1, 5)
    true_vals_tests = np.arcsin(x_tests)
    # Używamy rozwinięcia dla n = 7 (dowolny dobry wybór)
    approx_vals_tests = funSeriesExpansion(7, x_tests)
    abs_err_tests = np.abs(approx_vals_tests - true_vals_tests)
    rel_err_tests = np.where(true_vals_tests != 0, abs_err_tests / np.abs(true_vals_tests) * 100, 0)
    
    # Wypisanie tabeli wyników przy użyciu funkcji wektorowych (bez pętli)
    print("\n{:^10} {:^15} {:^15} {:^15}".format("x", "Rozwinięcie", "Abs error", "Rel error [%]"))
    
    # Utwórz macierz danych do wydruku
    table_data = np.column_stack((x_tests, approx_vals_tests, abs_err_tests, rel_err_tests))
    
    # Użyj np.savetxt z opcją fmt do formatowania i wydruku do StringIO
    output = io.StringIO()
    np.savetxt(output, table_data, fmt=('%10.5f', '%15.8f', '%15.8f', '%15.8f'))
    
    # Wydrukuj wynik
    print(output.getvalue())

# Wywołanie procedury testującej
testSeriesRandom()