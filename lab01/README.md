# Lab01 – Rozwinięcie funkcji w szereg (arcsin)

## Cel
Implementacja rozwinięcia funkcji \( \arcsin(x) \) w szereg Maclaurina (forma potęgowa) oraz analiza jakości przybliżenia w zależności od liczby wyrazów \( n \).

## Zawartość
- Funkcja `funSeriesExpansion(n, x)` – oblicza sumę szeregu do wyrazu indeksowanego `n` (adaptacja wzoru na rozwinięcie arcsin).
- Tabela błędów bezwzględnych i względnych dla rosnącego `n` (0..10) przy stałym `x=0.5`.
- Losowe testy dla kilku wartości `x∈[0,1]`.
- Procedura wyznaczająca minimalne `n` spełniające zadane kryterium błędu dla siatki punktów `x` (porównanie błędu względnego i bezwzględnego).
- Wykresy: porównanie funkcji dokładnej i przybliżeń dla wybranych `n` oraz mapa zależności „minimalne n vs x”.

## Instrukcja uruchomienia
```
python main.py
```
Skrypt generuje wykresy i wypisuje zestawienia błędów w konsoli.

## Uwagi
- W obliczeniach zastosowano wektoryzację `numpy` oraz kontrolę ostrzeżeń (np. dzielenie przez zero) przez `np.errstate`.
- Dla x bliskich 1 wymagane jest większe `n` dla osiągnięcia tego samego błędu.
