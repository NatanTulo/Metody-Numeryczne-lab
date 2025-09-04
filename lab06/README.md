# Lab06 – Aproksymacja odpowiedzi skokowej / identyfikacja parametrów

## Cel
Dopasowanie parametrów modelu dynamicznego (drugi rząd + zero) do zmierzonej odpowiedzi skokowej oraz porównanie rozwiązania numerycznego i „analitycznego” (metoda residuów / postać zamknięta).

## Zawartość
- Wczytanie danych (`data9.txt`).
- Parametry modelu: wzmocnienie `k`, stała czasowa `τ`, współczynnik tłumienia `ζ`, zero `τz`.
- Funkcja kosztu: suma kwadratów błędów między sygnałem a modelem.
- Optymalizacja: `scipy.optimize.fmin` (Nelder–Mead) z ograniczeniami miękkimi (filtr parametrów – zwracanie `inf`).
- Obsługa trzech przypadków tłumienia: niedotłumiony, krytyczny, przetłumiony.
- Porównanie przebiegów: dane, model numeryczny, postać analityczna.

## Uruchomienie
```
python main.py
```

## Uwagi
- Dla lepszej stabilności można użyć `least_squares` (Jacobian) lub skalowania osi czasu.
- Zastosowanie gradientu (np. BFGS) wymagałoby jawnych pochodnych – tu pominięto.
