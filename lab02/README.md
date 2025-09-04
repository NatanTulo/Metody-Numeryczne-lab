# Lab02 – Przybliżenie pochodnej (różnica centralna)

## Cel
Numeryczne przybliżenie pochodnej funkcji \( f(x)=\arcsin(x) \) w punkcie `x0=0.5` różnicą centralną oraz analiza zachowania błędu przy zmniejszaniu kroku `dx`.

## Zawartość
- Funkcja `funDerivativeApprox(x, dx, fun)` – przybliżenie pochodnej: \( (f(x+dx)-f(x-dx))/(2dx) \).
- Generowanie malejącej sekwencji kroków `dx = dx0 / 5^k`.
- Wyznaczenie „optymalnego” kroku minimalizującego błąd (kompromis: błąd obcięcia vs. błąd zaokrągleń).
- Wykres log–log: `dx` vs. błąd bezwzględny.
- Drugi wykres dla rozkładu przybliżonej pochodnej w zakresie `x∈[best_dx, 1-best_dx]`.

## Uruchomienie
```
python main.py
```

## Uwagi
- Dla bardzo małych `dx` błąd rośnie – dominują błędy numeryczne (katastrofa odejmowania i precyzja zmiennoprzecinkowa).
- Możliwe rozszerzenie: porównanie z przybliżeniem wyższego rzędu (np. 4‑punktowym).
