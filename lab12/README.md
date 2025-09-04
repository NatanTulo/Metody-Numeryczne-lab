# Lab12 – Zagadnienie brzegowe 2D (równanie przewodnictwa / Laplace + warunki mieszane)

## Cel
Rozwiązanie stanu ustalonego rozkładu temperatury w kwadratowej domenie z liniowo zmiennymi temperaturami na krawędziach oraz efektem konwekcyjnym (termin `h'` w równaniu) – metoda różnic skończonych.

## Model
- Siatka: `(Nx, Ny) = (11,11)` na prostokącie `Lx=Ly=20` (jednakowy krok). 
- Węzły brzegowe przypisane przez funkcje liniowe: górna i lewa krawędź 400→...; dolna/prawa 300→... (malejąco).
- Wewnętrzne równanie dyskretyzowane: pięciopunktowy schemat Laplace’a rozszerzony o człon konwekcyjny `coeff = h' * Δ^2`.
- Konstrukcja macierzy układu liniowego `A x = b` (indeksowanie funkcją pomocniczą `idx`).

## Zawartość
- Generacja macierzy `A` i prawej strony `b` z uwzględnieniem warunków Dirichleta.
- Rozwiązanie układu przez `np.linalg.solve`.
- Odwzorowanie wektora rozwiązania na pełną siatkę temperatur.
- Wizualizacja: mapa kolorów + naniesione fizyczne pozycje węzłów.
- Wypis próbek temperatur wewnętrznych.

## Uruchomienie
```
python main.py
```

## Uwagi
- Współczynnik konwekcyjny można parametryzować – aktualnie stały.
- Rozszerzenia: metoda iteracyjna (Gauss–Seidel / SOR), siatka nieregularna, eksport do formatu CSV.
