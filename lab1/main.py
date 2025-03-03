import numpy as np
import matplotlib.pyplot as plt

def funSeriesExpansion(n, x):
    x = np.array(x)  # konwersja do numpy array
    
    # Obsługa przypadku n=0 - zwróć x
    if n == 0:
        return x
    
    m = np.arange(n)
    max_index = 2*(n-1)
    # obliczenie silni od 0! do (2*(n-1))!
    fac = np.concatenate(([1], np.cumprod(np.arange(1, max_index+1))))
    
    # Bezpieczne obliczanie współczynników
    with np.errstate(divide='ignore', invalid='ignore'):
        coef = np.where(m == 0, 1.0, fac[2*m] / (4**m * (fac[m]**2) * (2*m + 1)))
    
    # Wszystkie NaN lub inf wartości zastępujemy zerami
    coef = np.nan_to_num(coef)
    
    if x.ndim == 0:  # x jest skalarem
        return np.sum(coef * x**(2*m+1))
    else:
        # Bezpieczne mnożenie wektorów z obsługą nieprawidłowych wartości
        expanded_x = x.reshape(-1, 1)
        expanded_powers = (2*m+1).reshape(1, -1)
        with np.errstate(invalid='ignore'):
            powers = np.power(expanded_x, expanded_powers)
        # Zastąp nieprawidłowe wartości zerami
        powers = np.nan_to_num(powers)
        return np.sum(coef * powers, axis=1)

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
for n_selected in [1, 5, 9]:
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
    
    # Bezpieczne obliczanie błędu względnego z użyciem np.where i np.errstate
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err_tests = np.where(true_vals_tests != 0, 
                                abs_err_tests / np.abs(true_vals_tests) * 100, 
                                0)
    # Zastąp nieskończone wartości i NaN zerami
    rel_err_tests = np.nan_to_num(rel_err_tests)
    
    # Wypisanie tabeli wyników w stylu pierwszej tabeli (używając f-string)
    print(f"\n{'x':>10} {'Rozwinięcie':>15} {'Błąd bezwzględny':>20} {'Błąd względny [%]':>20}")
    
    # Utwórz macierz danych do wydruku
    table_data = np.column_stack((x_tests, approx_vals_tests, abs_err_tests, rel_err_tests))
    
    # Formatowanie z wykorzystaniem funkcji map i join (bez jawnych pętli)
    formatted_rows = map(lambda row: f"{row[0]:10.5f} {row[1]:15.8f} {row[2]:20.8f} {row[3]:20.8f}", table_data)
    print('\n'.join(formatted_rows))

# Wywołanie procedury testującej
testSeriesRandom()

def find_min_n_for_error_threshold(epsilon=1e-3, use_relative_error=True):
    """
    Wyznacza minimalne n dla różnych wartości x, aby błąd był mniejszy niż epsilon.
    Używa operacji wektoryzowanych zamiast jawnych pętli po x.
    """
    # Generowanie równomiernie rozłożonych wartości x
    x_values = np.linspace(0, 1, 100)
    true_values = np.arcsin(x_values)
    
    # Maksymalne n do sprawdzenia
    max_n = 30
    
    # Pre-obliczenie wszystkich rozwinięć dla n od 0 do max_n dla wszystkich x
    all_expansions = np.zeros((len(x_values), max_n + 1))
    for n in range(max_n + 1):
        all_expansions[:, n] = funSeriesExpansion(n, x_values)
    
    # Obliczenie błędów
    abs_errors = np.abs(all_expansions - true_values.reshape(-1, 1))
    
    if use_relative_error:
        # Bezpieczne obliczenie błędu względnego bez ostrzeżeń
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_errors = abs_errors / np.abs(true_values.reshape(-1, 1))
            errors = np.where(np.isfinite(raw_errors), raw_errors, abs_errors)
    else:
        errors = abs_errors
    
    # Znajdź minimalne n dla każdego x (bez pętli)
    mask = errors < epsilon
    # Dla każdego wiersza znajdź indeks pierwszej wartości True
    min_n_indices = np.argmax(mask, axis=1)
    
    # Jeśli dla jakiegoś x żadne n nie spełnia warunku, ustaw max_n
    min_n_indices = np.where(np.any(mask, axis=1), min_n_indices, max_n)
    
    return x_values, min_n_indices

def plot_min_n_dependency():
    """
    Rysuje wykres zależności minimalnego n od wartości x, 
    dla którego błąd < epsilon
    """
    epsilon = 1e-3  # zadana granica błędu
    
    # Wyznacz minimalne n dla błędu względnego i bezwzględnego
    x_rel, min_n_rel = find_min_n_for_error_threshold(epsilon, use_relative_error=True)
    x_abs, min_n_abs = find_min_n_for_error_threshold(epsilon, use_relative_error=False)
    
    # Tworzenie wykresu
    plt.figure(figsize=(12, 6))
    plt.plot(x_rel, min_n_rel, 'o-', markersize=4, label="Błąd względny")
    plt.plot(x_abs, min_n_abs, 's-', markersize=4, label="Błąd bezwzględny")
    plt.xlabel("Wartość x")
    plt.ylabel("Minimalne n")
    plt.title(f"Minimalne n potrzebne dla błędu < {epsilon}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Wywołanie funkcji wykreślającej zależność minimalnego n od x
plot_min_n_dependency()