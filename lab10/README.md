# Lab10 – Rozwiązywanie równania różniczkowego (Euler / Heun / Midpoint)

## Cel
Porównanie prostych metod jawnych pierwszego rzędu (Euler) i ulepszonych (Heun, punkt środkowy) z analitycznym rozwiązaniem nietrywialnego równania \( y'(t) = (1 - 4t^3) \sqrt{y} \) przy warunku początkowym \( y(0)=2 \).

## Zawartość
- Walidacja dziedziny: wymuszenie nieujemności (przycięcie do zera) by uniknąć \( \sqrt{y<0} \).
- Automatyczne wyznaczenie maksymalnego `t` zapewniającego dodatni wyraz pod pierwiastkiem w rozwiązaniu analitycznym.
- Interaktywne pobranie czasu końcowego `tk`.
- Wykres porównawczy wszystkich trajektorii.

## Uruchomienie
```
python main.py
```

## Uwagi
- Przy większym kroku różnice między metodami rosną – można dodać analizę błędu globalnego vs. krok.
- Metody jawne mogą tracić stabilność dla ostrzejszych nieliniowości – tu łagodzone przez ograniczenie dziedziny.
