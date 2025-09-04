# Lab05 – Sterowanie optymalne (LQR) dla układu dyskretnego

## Cel
Analiza dyskretnego układu opisanego transmitancją oraz wyprowadzenie modelu przestrzeni stanów; zaprojektowanie regulatora LQR i zbadanie wpływu wag macierzy \( Q \) i \( R \).

## Zawartość
- Transmitancja: \( G(z) = (z^2 + z + 1) / (z^3 - 3.2 z^2 + 2.75 z - 0.65) \).
- Konwersja do postaci kanonicznej przestrzeni stanów (macierze A, B, C, D).
- Odpowiedzi impulsowa i skokowa (biblioteka `scipy.signal`).
- LQR: rozwiązanie równania dyskretnego Riccatiego (`solve_discrete_are`) oraz iteracyjna aproksymacja macierzy \( P \).
- Sprzężenie zwrotne (A - B K) i porównanie odpowiedzi.
- Animacja wpływu zmian wag (czasowy przebieg odpowiedzi dla dynamicznie modyfikowanych parametrów `c1`, `c2`).

## Uruchomienie
```
python main.py
```

## Uwagi
- Bieguny układu weryfikowane pod kątem stabilności (|z| < 1).
- Dobór wag wpływa na kompromis: szybkość vs. „koszt sterowania”.
- Możliwe rozszerzenie: dyskretyzacja z ciągłego modelu oraz porównanie obu form.
