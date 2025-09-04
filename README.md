# Metody Numeryczne – Zestaw Laboratoriów

Repozytorium zawiera rozwiązania kolejnych laboratoriów z metod numerycznych. Każdy folder `labXX` posiada skrypt `main.py` (i ewentualne dane) ilustrujący wybrane zagadnienia: szeregi potęgowe, aproksymację pochodnych, układy równań liniowych i nieliniowych, sterowanie optymalne (LQR), aproksymację i estymację parametrów, filtr Kalmana, interpolację, całkowanie numeryczne, metody rozwiązywania równań różniczkowych zwyczajnych oraz zagadnienia brzegowe dla równania przewodnictwa.

## Przegląd laboratoriów
| Lab | Temat (skrót) | Główne elementy |
|-----|---------------|-----------------|
| 01 | Szereg Maclaurina / rozwinięcie funkcji | Rozwinięcie arcsin(x), analiza błędu, minimalne n dla zadanego progu |
| 02 | Przybliżenie pochodnej | Różnica centralna, dobór optymalnego kroku, skala log–log |
| 03 | Układy równań liniowych | LU z częściowym wyborem, macierz odwrotna, analiza kontrybucji źródeł |
| 04 | Równania nieliniowe | Iteracyjne podstawianie, bisekcja, Newton (SciPy + własna), wielokrotne pierwiastki |
| 05 | Sterowanie optymalne | Model dyskretny, LQR, Riccati iteracyjnie, animacja wpływu wag Q/R |
| 06 | Aproksymacja / identyfikacja | Dopasowanie parametrów odpowiedzi skokowej (fmin), metoda residuów, porównanie numeryczne vs analityczne |
| 07 | Filtr Kalmana | Model ruchu 2D, dwa warianty inicjalizacji, predykcja trajektorii |
| 08 | Interpolacja | Wielomian Newtona, splajn kubiczny (SciPy i własny), porównanie | 
| 09 | Całkowanie | Trapezy + oszacowanie błędu, Romberg, Gauss-Legendre 3‑punktowy, porównanie z całką analityczną |
| 10 | Równanie różniczkowe ODE (1D) | Metody: Euler, Heun, punkt środkowy + rozwiązanie analityczne i ograniczenie dziedziny |
| 11 | Układ ODE (Lorenz) | `solve_ivp` (RK45) vs własny RK4, trajektorie czasowe i fazowe 3D |
| 12 | Zagadnienie brzegowe 2D | Równanie przewodnictwa (Laplace + konwekcja) metodą różnic skończonych, rozwiązanie układu liniowego |

## Wymagania
Typowo wykorzystywane biblioteki:
- `numpy`
- `matplotlib`
- `scipy` (optimize, signal, interpolate, integrate, linalg)

Instalacja przykładowa (wspólne środowisko):
```
python -m venv .venv
./.venv/Scripts/Activate.ps1  # Windows PowerShell
pip install numpy matplotlib scipy
```

## Uruchamianie
Wejdź do wybranego katalogu i uruchom:
```
python main.py
```
Niektóre skrypty oczekują obecności plików danych (`data7.txt`, `data9.txt`) – znajdują się one w odpowiadających katalogach.

## Konwencje i założenia
- Brak punktacji w README (nie rekonstruujemy ocen z materiału źródłowego).
- Opis każdego laboratorium streszcza zarówno intencję zadania (na podstawie pliku `tematy`) jak i faktyczną implementację w kodzie.
- W razie potrzeby parametry (np. kroki, liczby iteracji) można modyfikować wprost w kodzie.

## Spis treści README dla poszczególnych laboratoriów
Każdy plik `labXX/README.md` zawiera: opis celu, zakres metod, instrukcję uruchomienia i krótkie uwagi (np. stabilność, ograniczenia rozwiązania, możliwe rozszerzenia).

---
Jeżeli chcesz dodać testy jednostkowe (np. dla porównań metod) lub automatyczną walidację błędów, można utworzyć dodatkowy moduł `tests/` z asercjami względem wyników znanych analitycznie (np. lab09, lab10).
