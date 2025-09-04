# Lab07 – Filtr Kalmana (trajektoria 2D)

## Cel
Estymacja trajektorii obiektu poruszającego się w 2D przy zaszumionych pomiarach położenia oraz porównanie wpływu różnych warunków początkowych prędkości na predykcję.

## Zawartość
- Model stanu: \([x, y, v_x, v_y]^T\) z krokiem czasowym T=1.
- Macierze: przejścia `A`, hałasu procesowego `Q` (poprzez `G` i wariancje), obserwacji `H`, kowariancji pomiaru `R`.
- Dwa scenariusze inicjalizacji: prędkość zerowa oraz prędkość wyliczona z pierwszych dwóch pomiarów.
- Predykcja pozycji na horyzoncie +5s bez aktualizacji pomiarowej.
- Wykresy: pomiary vs trajektoria filtrowana + odcinek predykcji.

## Uruchomienie
```
python main.py
```
Upewnij się, że plik `data7.txt` znajduje się w folderze.

## Uwagi
- Możliwe rozszerzenie: adaptacyjna estymacja `Q` i `R` lub filtr rozszerzony (EKF) dla modelu nieliniowego.
