# 🧪 SMA Maze Solver — Slime Mould Algorithm

**Przedmiot:** Głębokie Nauczanie  
**Algorytm:** Slime Mould Algorithm (SMA) — Wei et al. (2024), Li et al. (2020)

## 📖 Opis projektu

Aplikacja webowa implementująca **Slime Mould Algorithm (SMA)** do znajdowania
najkrótszej ścieżki w labiryncie. Algorytm inspirowany jest zachowaniem śluzowca
*Physarum polycephalum* podczas żerowania — organizm tworzy sieć żyłek łączących
źródła pokarmu, optymalizując trasę za pomocą mechanizmu sprzężenia zwrotnego.

### Funkcjonalności:
- 🐭 **10 predefiniowanych labiryntów** (od 11×11 do 51×51) + generator własnych
- 🧬 **Algorytm SMA** z konfigurowalnymi parametrami
- 📊 **Porównanie z A\*, Dijkstra, BFS** — ścieżki, czasy, metryki
- 🌐 **GUI w przeglądarce** — interaktywna wizualizacja z przełączaniem ścieżek
- 📉 **Krzywa zbieżności** SMA w czasie rzeczywistym
- 🐳 **Docker** — uruchomienie jednym poleceniem

## 🚀 Uruchomienie

### Docker (rekomendowane):
```bash
docker-compose up --build
```
Aplikacja dostępna pod: **http://localhost:5000**

### Bez Dockera:
```bash
pip install -r requirements.txt
python app.py
```

## 🧬 Algorytm SMA — Model matematyczny

### Aktualizacja pozycji (Eq. 1):
```
X(t+1) = { rand*(UB-LB) + LB,                  jeśli rand < z    (eksploracja losowa)
          { Xb(t) + vb*(W*XA(t) - XB(t)),       jeśli r < p       (eksploatacja)
          { vc * X(t),                            jeśli r >= p      (kontrakcja)
```

### Parametry:
| Parametr | Opis | Domyślna wartość |
|----------|------|------------------|
| `N` | Rozmiar populacji (liczba agentów) | 50 |
| `max_t` | Maksymalna liczba iteracji | 200 |
| `z` | Prawdopodobieństwo losowej eksploracji | 0.03 |
| `p` | `tanh\|S(i) - DF\|` — fitness-based probability | obliczane |
| `a` | `arctanh(1 - t/max_t)` — granica adaptacyjna | obliczane |
| `vb` | Parametr oscylacji ∈ [-a, a] | losowy |
| `vc` | Parametr kontrakcji (1→0 liniowo) | obliczany |
| `W` | Waga na podstawie rankingu fitness | obliczana (Eq. 4) |

### Adaptacja do labiryntu:
- **Agent** = kandydacka ścieżka (sekwencja komórek od startu do celu)
- **Fitness** = długość ścieżki + kara za niedotarcie do celu
- **Eksploracja** = generowanie nowej losowej ścieżki
- **Eksploatacja** = krzyżowanie (crossover) najlepszych ścieżek
- **Kontrakcja** = mutacja (re-routing fragmentu ścieżki)

## 📊 Porównywane algorytmy

| Algorytm | Typ | Optymalność | Złożoność |
|----------|-----|-------------|-----------|
| **SMA** | Metaheurystyczny | Przybliżony | O(N × max_t × L) |
| **A\*** | Deterministyczny | Optymalny | O(b^d) |
| **Dijkstra** | Deterministyczny | Optymalny | O(V + E log V) |
| **BFS** | Deterministyczny | Optymalny (bez wag) | O(V + E) |

## 📁 Struktura projektu

```
sma-maze-solver/
├── app.py                 # Flask web server
├── sma_algorithm.py       # Implementacja SMA
├── classic_algorithms.py  # A*, Dijkstra, BFS
├── maze_generator.py      # Generatory labiryntów (DFS, Prim, Open)
├── templates/
│   └── index.html         # GUI webowe
├── requirements.txt       # Zależności Python
├── Dockerfile             # Obraz Docker
├── docker-compose.yml     # Docker Compose
└── README.md              # Dokumentacja
```

## 📚 Bibliografia

1. Li, S.; Chen, H. *Slime mould algorithm: A new method for stochastic optimization.* Future Gener. Comput. Syst. 2020, 111, 300–323.
2. Wei, Y.; Othman, Z.; Daud, K.M.; Luo, Q.; Zhou, Y. *Advances in Slime Mould Algorithm: A Comprehensive Survey.* Biomimetics 2024, 9, 31.
