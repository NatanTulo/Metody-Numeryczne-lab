# Lab03 – Układy równań liniowych i analiza źródeł emisji

## Cel
Rozwiązanie układu równań opisującego transport (dyspersję) zanieczyszczeń / stężeń w kilku połączonych pomieszczeniach oraz dekompozycja udziału źródeł w wybranym węźle.

## Zawartość
- Budowa macierzy `A` oraz wektora `b` na podstawie przepływów i źródeł.
- Własna implementacja faktoryzacji LU z częściowym wyborem elementu głównego (`lu_decomposition`).
- Rozwiązanie: najpierw `L y = P^T b`, potem `U c = y`.
- Obliczenie macierzy odwrotnej `A^{-1}` kolumna po kolumnie (rozwiązywanie układu dla wektorów bazowych).
- Analiza superpozycji: osobne wektory wymuszeń dla trzech źródeł – udział procentowy w pokoju dziecka.
- Study case: zmiana intensywności emisji i przeliczenie wyników.

## Uruchomienie
```
python main.py
```

## Uwagi
- Implementacja LU nie zawiera skalowania – przy silnie źle uwarunkowanych macierzach należałoby rozszerzyć o pełny pivoting.
- Wektory emisji są zdefiniowane manualnie – można dodać moduł wczytujący scenariusze z pliku.
