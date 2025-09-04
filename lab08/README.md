# Lab08 – Interpolacja: Wielomian Newtona i splajn kubiczny

## Cel
Porównanie metod interpolacji trajektorii: wielomian Newtona (różnice dzielone) oraz splajn kubiczny (SciPy i implementacja własna typu „natural”).

## Zawartość
- Wczytanie punktów (x, y) z `data9.txt`.
- Funkcje: `divided_differences`, `newton_eval`.
- Splajn naturalny: własna konstrukcja układu trójdiagonalnego dla drugich pochodnych (`c`) + ocena w punktach.
- Porównanie z `scipy.interpolate.CubicSpline`.
- Wydruk różnic między implementacjami (sprawdzenie poprawności).

## Uruchomienie
```
python main.py
```

## Uwagi
- Wielomian wysokiego stopnia może ulegać efektowi Rungego – splajny zwykle bardziej stabilne.
- Można dodać warianty warunków brzegowych (clamped / not-a-knot).
