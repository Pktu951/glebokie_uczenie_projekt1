import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators

os.makedirs("data_tests", exist_ok=True)

print("=== TESTY bibliotek oraz datasetów ===")

# NUMPY
print("=== NUMPY ===")
arr = np.array([1, 2, 3, 4, 5])
print(f"Tablica: {arr}")
print(f"Srednia: {arr.mean()}")

# PANDAS
print("\n=== PANDAS ===")
df = pd.DataFrame({
    "algorytm": ["Dijkstra", "SMA"],
    "czas": [0.5, 1.2],
    "dlugosc_sciezki": [15, 15]
})
print(df)

# MATPLOTLIB
print("\n=== MATPLOTLIB ===")
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Test wykresu")
plt.savefig("data_tests/test_wykres.png")
print("Wykres zapisany do data_tests/test_wykres.png")

# OPENCV
print("\n=== OPENCV ===")
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (90, 90), (0, 255, 0), 2)
cv2.imwrite("data_tests/test_obraz.png", img)
print("Obraz zapisany do data_tests/test_obraz.png")

# PYTORCH
print("\n=== PYTORCH ===")
tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"Tensor: {tensor}")
print(f"Suma: {tensor.sum()}")

# MAZE DATASET
print("\n=== MAZE DATASET ===")
cfg = MazeDatasetConfig(
    name="test",
    grid_n=5,
    n_mazes=3,
    maze_ctor=LatticeMazeGenerators.gen_dfs,
)
dataset = MazeDataset.from_config(cfg, local_base_path="data_tests")

print(f"Wygenerowano {len(dataset)} labiryntow")
print(f"Pierwszy labirynt: {dataset[0]}")

print("\n ----------- Jeśli to widzisz to jest git xd -----------")