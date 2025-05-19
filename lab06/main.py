import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import math

# Wczytaj dane z pliku
data = np.loadtxt("data9.txt")
t = data[:, 0]  # Punkty czasu
h_measured = data[:, 1]  # Zmierzona odpowiedź skokowa

# Zdefiniuj analityczną postać odpowiedzi skokowej dla danej funkcji transmitancji
def step_response(t, params):
    k, tau, zeta, tau_z = params
    
    omega_n = 1 / tau
    
    # Obsłuż przypadek, gdy ζ = 1 (krytyczne tłumienie)
    if abs(zeta - 1) < 1e-6:
        h = k * (1 - (1 + t/tau) * np.exp(-t/tau))
    # Obsłuż przypadek, gdy ζ < 1 (niedotłumiony)
    elif zeta < 1:
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        phi = np.arctan(zeta / np.sqrt(1 - zeta**2))
        exp_term = np.exp(-zeta * omega_n * t)
        h = k * (1 - exp_term / np.sin(phi) * np.sin(omega_d * t + phi))
    # Obsłuż przypadek, gdy ζ > 1 (przetłumiony)
    else:
        s1 = -omega_n * (zeta + np.sqrt(zeta**2 - 1))
        s2 = -omega_n * (zeta - np.sqrt(zeta**2 - 1))
        h = k * (1 - (s2*np.exp(s1*t) - s1*np.exp(s2*t))/(s2 - s1))
    
    # Zastosuj efekt zera przy -1/τz
    if tau_z > 0:
        h = h * (1 + tau_z * np.gradient(h, t))
    
    return h

# Zdefiniuj funkcję błędu (suma kwadratów błędów)
def error_function(params):
    k, tau, zeta, tau_z = params
    # Upewnij się, że parametry są fizycznie uzasadnione
    if k <= 0 or tau <= 0 or zeta <= 0 or tau_z < 0:
        return float('inf')
    
    # Oblicz teoretyczną odpowiedź skokową
    h_theoretical = step_response(t, params)
    
    # Oblicz sumę kwadratów błędów
    sse = np.sum((h_measured - h_theoretical)**2)
    
    return sse

# Początkowe oszacowanie parametrów [k, tau, ζ, τz]
initial_params = [1.0, 1.0, 0.5, 0.1]

# Znajdź optymalne parametry za pomocą fmin
print("Optymalizacja parametrów...")
optimal_params = fmin(error_function, initial_params, maxiter=10000, maxfun=10000, disp=True)

# Wypisz optymalne parametry
k_opt, tau_opt, zeta_opt, tau_z_opt = optimal_params
print(f"Parametry optymalne:")
print(f"k = {k_opt:.4f}")
print(f"τ = {tau_opt:.4f}")
print(f"ζ = {zeta_opt:.4f}")
print(f"τz = {tau_z_opt:.4f}")

# Oblicz teoretyczną odpowiedź skokową z optymalnymi parametrami
h_theoretical = step_response(t, optimal_params)

# Wykreśl wyniki
plt.figure(figsize=(10, 6))
plt.plot(t, h_measured, 'bo', label='Dane zmierzone')
plt.plot(t, h_theoretical, 'r-', label='Odpowiedź teoretyczna')
plt.xlabel('Czas (s)')
plt.ylabel('Amplituda')
plt.title('Odpowiedź skokowa - Dane zmierzone vs odpowiedź teoretyczna')
plt.legend()
plt.grid(True)
plt.show()

# Metoda analityczna oparta o rozkład ułamków prostych - metoda residuów. 
def analytical_step_response(t, params):
    k, tau, zeta, tau_z = params
    omega_n = 1 / tau
    if abs(zeta - 1) < 1e-6:
        h = k * (1 - (1 + t/tau) * np.exp(-t/tau))
    elif zeta < 1:
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        phi = np.arctan(zeta / np.sqrt(1 - zeta**2))
        exp_term = np.exp(-zeta * omega_n * t)
        h = k * (1 - exp_term / np.sin(phi) * np.sin(omega_d * t + phi))
    else:
        s1 = -omega_n * (zeta + np.sqrt(zeta**2 - 1))
        s2 = -omega_n * (zeta - np.sqrt(zeta**2 - 1))
        h = k * (1 - (s2 * np.exp(s1*t) - s1 * np.exp(s2*t)) / (s2 - s1))
    
    if tau_z > 0:
        h = h * (1 + tau_z * np.gradient(h, t))
    return h

# Oblicz analityczną odpowiedź skokową z optymalnymi parametrami
h_analytical = analytical_step_response(t, optimal_params)

# Wykreśl wyniki z rozwiązaniem analitycznym
plt.figure(figsize=(10, 6))
plt.plot(t, h_measured, 'bo', label='Dane zmierzone')
plt.plot(t, h_theoretical, 'r-', label='Rozwiązanie numeryczne')
plt.plot(t, h_analytical, 'g--', label='Rozwiązanie analityczne')
plt.xlabel('Czas (s)')
plt.ylabel('Amplituda')
plt.title('Odpowiedź skokowa - Dane zmierzone vs odpowiedź numeryczna i analityczna')
plt.legend()
plt.grid(True)
plt.show()