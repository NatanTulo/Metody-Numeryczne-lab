import numpy as np
import matplotlib.pyplot as plt

def funSeriesExpansion(n, x):
    x = np.array(x)
    
    if n == 0:
        return x
    
    m = np.arange(n)
    max_index = 2*(n-1)
    # obliczenie silni
    fac = np.concatenate(([1], np.cumprod(np.arange(1, max_index+1))))
    
    # zabezpieczenie przed dzieleniem przez zero
    with np.errstate(divide='ignore', invalid='ignore'):
        coef = np.where(m == 0, 1.0, fac[2*m] / (4**m * (fac[m]**2) * (2*m + 1)))
    
    coef = np.nan_to_num(coef)
    
    if x.ndim == 0:  # x jest skalarem
        return np.sum(coef * x**(2*m+1))
    else:
        expanded_x = x.reshape(-1, 1)
        expanded_powers = (2*m+1).reshape(1, -1)
        with np.errstate(invalid='ignore'):
            powers = np.power(expanded_x, expanded_powers)
        powers = np.nan_to_num(powers)
        return np.sum(coef * powers, axis=1)

# Porównanie wartości rozwinięcia z funkcją arcsin(x)
x = 0.5
true_val = np.arcsin(x)
print(f"{'n':>3} {'Rozwinięcie':>15} {'Błąd bezwzględny':>20} {'Błąd względny [%]':>20}")
for n in range(11):
    approx = funSeriesExpansion(n, x)
    abs_err = abs(approx - true_val)
    rel_err = (abs_err/abs(true_val)*100) if true_val != 0 else 0
    print(f"{n:3d} {approx:15.8f} {abs_err:20.8f} {rel_err:20.8f}")

# Wykres porównawczy
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
    x_tests = np.random.uniform(0, 1, 5)
    true_vals_tests = np.arcsin(x_tests)
    approx_vals_tests = funSeriesExpansion(7, x_tests)
    abs_err_tests = np.abs(approx_vals_tests - true_vals_tests)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err_tests = np.where(true_vals_tests != 0, 
                                abs_err_tests / np.abs(true_vals_tests) * 100, 
                                0)
    rel_err_tests = np.nan_to_num(rel_err_tests)
    
    print(f"\n{'x':>10} {'Rozwinięcie':>15} {'Błąd bezwzględny':>20} {'Błąd względny [%]':>20}")
    
    table_data = np.column_stack((x_tests, approx_vals_tests, abs_err_tests, rel_err_tests))
    
    formatted_rows = map(lambda row: f"{row[0]:10.5f} {row[1]:15.8f} {row[2]:20.8f} {row[3]:20.8f}", table_data)
    print('\n'.join(formatted_rows))

testSeriesRandom()

def find_min_n_for_error_threshold(epsilon=1e-3, use_relative_error=True):
    """Wyznacza minimalne n dla różnych x, aby błąd był mniejszy niż epsilon"""
    x_values = np.linspace(0, 1, 100)
    true_values = np.arcsin(x_values)
    
    max_n = 30
    
    # obliczenie wszystkich rozwinięć
    all_expansions = np.zeros((len(x_values), max_n + 1))
    for n in range(max_n + 1):
        all_expansions[:, n] = funSeriesExpansion(n, x_values)
    
    abs_errors = np.abs(all_expansions - true_values.reshape(-1, 1))
    
    if use_relative_error:
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_errors = abs_errors / np.abs(true_values.reshape(-1, 1))
            errors = np.where(np.isfinite(raw_errors), raw_errors, abs_errors)
    else:
        errors = abs_errors
    
    # znajdź minimalne n spełniające warunek błędu
    mask = errors < epsilon
    min_n_indices = np.argmax(mask, axis=1)
    min_n_indices = np.where(np.any(mask, axis=1), min_n_indices, max_n)
    
    return x_values, min_n_indices

def plot_min_n_dependency():
    """Wykres zależności minimalnego n od x dla zadanego epsilon"""
    epsilon = 1e-3
    
    x_rel, min_n_rel = find_min_n_for_error_threshold(epsilon, use_relative_error=True)
    x_abs, min_n_abs = find_min_n_for_error_threshold(epsilon, use_relative_error=False)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_rel, min_n_rel, 'o-', markersize=4, label="Błąd względny")
    plt.plot(x_abs, min_n_abs, 's-', markersize=4, label="Błąd bezwzględny")
    plt.xlabel("Wartość x")
    plt.ylabel("Minimalne n")
    plt.title(f"Minimalne n potrzebne dla błędu < {epsilon}")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_min_n_dependency()