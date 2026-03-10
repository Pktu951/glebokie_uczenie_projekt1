# Maze Pathfinding: Advances in Slime Mould Algorithm

> Student project for **Deep learning and Computational intelligence** courses.

Comparison of two pathfinding approaches in maze environments:
- **Dijkstra's Algorithm** — classic, deterministic, always finds the optimal path
- **Slime Mould Algorithm (SMA)** — bio-inspired metaheuristic

> Reference: Wei et al. (2024). *Advances in Slime Mould Algorithm: A Comprehensive Survey*. Biomimetics, 9(1), 31. https://doi.org/10.3390/biomimetics9010031

---

## Project Structure

```
maze/
├── Dockerfile
├── requirements.txt
├── mazes/          
├── results/        
├── data_tests/     # test outputs
├── src/            # main folder
│   └── main.py
└── tests/          # tests
    ├── test_main.py
    ├── test_dijkstra.py
    ├── test_sma.py
    └── test_maze_loader.py
```

---

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

## Setup & Running

### 1. Build the Docker image
```powershell
docker build -t maze .
```

### 2. Run a file
```powershell
docker run -v C:\Users\your-username\Desktop\maze:/app maze python src/main.py
```

### 3. Run tests
```powershell
docker run -v C:\Users\your-username\Desktop\maze:/app maze pytest tests/
```

---

## Tech Stack

| Library | Version |
|---------|---------|
| Python | 3.13 |
| PyTorch | 2.6.0+cpu |
| NumPy | 2.2.3 |
| Pandas | 2.2.3 |
| OpenCV | 4.11.0 |
| Matplotlib | 3.10.1 |
| maze-dataset | 1.3.0 |

---


### Full requirements.txt

```
numpy==2.2.3
pandas==2.2.3
Pillow==11.1.0
opencv-python-headless==4.11.0.86
matplotlib==3.10.1
seaborn==0.13.2
scikit-learn==1.6.1
joblib==1.4.2
tqdm==4.67.1
loguru==0.7.3
pytest==8.3.5
pytest-cov==6.0.0
maze-dataset==1.3.0
muutils==0.8.3
torch==2.6.0+cpu
torchvision==0.21.0+cpu
torchaudio==2.6.0+cpu
```

## Status



---

## Future Work

- Dijkstra's algorithm implementation
- Slime Mould Algorithm implementation
- Comparison and visualization of all approaches

---

## Author

Student project — Deep learning and Computational intelligence, 2026

```
Łukasz Mamczura
Julia Wiater
Dominik Gwardzik
Filip Mazur
Kacper Wójcik
```