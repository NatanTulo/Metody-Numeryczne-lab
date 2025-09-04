# Lab11 – Układ równań różniczkowych (Lorenz) – solve_ivp vs. własny RK4

## Cel
Porównanie implementacji integratora Rungego–Kutty 4. rzędu z dostępną w `scipy.integrate.solve_ivp` (RK45) dla klasycznego chaotycznego układu Lorenza.

## Zawartość
- Definicja układu (σ=10, ρ=28, β=8/3).
- Integracja: krok stały (RK4) i adaptacyjny (RK45, parametry tolerancji).
- Porównanie końcowych wartości i maksymalnych różnic komponentów.
- Wykresy: przebiegi x(t), y(t), z(t) + dwie osobne trajektorie fazowe 3D + wykres porównawczy.
- Proste statystyki zakresów zmiennych.

## Uruchomienie
```
python main.py
```

## Uwagi
- Ze względu na wrażliwość układu na warunki początkowe rozbieżność wzrasta w czasie – normalne zachowanie w dynamice chaotycznej.
- Możliwe rozszerzenie: analiza Lyapunova lub adaptacja kroku w implementacji własnej.
