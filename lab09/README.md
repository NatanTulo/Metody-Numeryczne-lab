# Lab09 – Całkowanie numeryczne i porównanie metod

## Cel
Porównanie różnych metod całkowania dla znanego wielomianu (znana całka analityczna) na przedziale \([a,b]\).

## Zawartość
- Całka analityczna przez całkowanie symboliczne „ręcznie” (sumowanie wkładów wielomianu).
- Metoda trapezów z estymacją błędu \(O(1/n^2)\) (wykorzystanie średniej drugiej pochodnej).
- Romberg: tablica ulepszanych przybliżeń (ekstrapolacja Richardsona) i zatrzymanie po spełnieniu progu błędu względnego.
- Kwadratura Gaussa–Legendre’a (3-punktowa) z transformacją przedziału.
- Tabela wyników: integral, błąd absolutny i względny.

## Uruchomienie
```
python main.py
```

## Uwagi
- Romberg zakończy się wyjątkiem jeśli wymagany próg błędu nie zostanie osiągnięty w zadanej liczbie iteracji – można rozszerzyć limit.
- Dodanie adaptacyjnej metody Simpsona stanowi naturalne uzupełnienie.
