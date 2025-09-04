# Lab04 – Równania nieliniowe (metody: iteracyjna, bisekcja, Newton)

## Cel
Wyznaczenie punktów przecięcia dwóch funkcji: \( f_1(x)=-3x^2-2x+4 \) oraz \( f_2(x)=-\frac{x^2}{1+2x} \) (z uwzględnieniem asymptoty w \( x=-0.5 \)).

## Metody
- Graficzna analiza i identyfikacja przybliżonych punktów startowych.
- Iteracyjne podstawianie: aktualizacja \( x_{k+1} = x_k + \alpha (f_2(x_k) - f_1(x_k)) \).
- Bisekcja – deterministyczna konwergencja przy prawidłowym przedziale.
- Newton–Raphson: wariant `scipy.optimize.newton` (adaptacyjny) i własna implementacja z ręcznie policzoną pochodną \( f'(x) \) dla \( f(x)=f_1(x)-f_2(x) \).
- Obsługa wielokrotnych rozwiązań (różne punkty startowe).

## Uruchomienie
```
python main.py
```

## Uwagi
- Funkcja \( f_2 \) nieskończona w pobliżu \( x=-0.5 \) – należy uważać na dobór przedziałów.
- Własna metoda Newtona sygnalizuje zatrzymanie przy małej pochodnej.
- Rozszerzenie: adaptacyjny dobór \( \alpha \) w iteracyjnych podstawieniach.
