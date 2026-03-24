"""
Maze Generator
==============
Multiple maze generation algorithms + preset mazes of various sizes.
0 = walkable, 1 = wall
"""

import numpy as np
import random
from typing import Tuple


def generate_dfs_maze(rows: int, cols: int, seed: int = None) -> np.ndarray:
    """
    Generate maze using Recursive Backtracking (DFS).
    Creates perfect mazes (exactly one path between any two points).
    """
    if seed is not None:
        random.seed(seed)

    # Ensure odd dimensions for proper maze structure
    rows = rows if rows % 2 == 1 else rows + 1
    cols = cols if cols % 2 == 1 else cols + 1

    maze = np.ones((rows, cols), dtype=int)

    def carve(r, c):
        maze[r, c] = 0
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 < nr < rows and 0 < nc < cols and maze[nr, nc] == 1:
                maze[r + dr // 2, c + dc // 2] = 0
                carve(nr, nc)

    carve(1, 1)

    # Ensure start and end are open
    maze[1, 0] = 0  # Entry
    maze[rows - 2, cols - 1] = 0  # Exit

    return maze


def generate_prim_maze(rows: int, cols: int, seed: int = None) -> np.ndarray:
    """
    Generate maze using Prim's algorithm.
    Creates mazes with more branching than DFS.
    """
    if seed is not None:
        random.seed(seed)

    rows = rows if rows % 2 == 1 else rows + 1
    cols = cols if cols % 2 == 1 else cols + 1

    maze = np.ones((rows, cols), dtype=int)
    walls = []

    def add_walls(r, c):
        for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            nr, nc = r + dr, c + dc
            if 0 < nr < rows and 0 < nc < cols:
                walls.append((r + dr // 2, c + dc // 2, nr, nc))

    start_r, start_c = 1, 1
    maze[start_r, start_c] = 0
    add_walls(start_r, start_c)

    while walls:
        idx = random.randint(0, len(walls) - 1)
        wr, wc, nr, nc = walls.pop(idx)

        if maze[nr, nc] == 1:
            maze[wr, wc] = 0
            maze[nr, nc] = 0
            add_walls(nr, nc)

    maze[1, 0] = 0
    maze[rows - 2, cols - 1] = 0

    return maze


def generate_open_maze(rows: int, cols: int, wall_density: float = 0.3, seed: int = None) -> np.ndarray:
    """
    Generate an open maze with random walls.
    Easier for algorithms - multiple possible paths.
    """
    if seed is not None:
        np.random.seed(seed)

    maze = np.zeros((rows, cols), dtype=int)

    # Add border walls
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    # Random internal walls
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if np.random.random() < wall_density:
                maze[r, c] = 1

    # Ensure start and end are open
    maze[1, 1] = 0
    maze[rows - 2, cols - 2] = 0

    # Clear a small area around start and end
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            sr, sc = 1 + dr, 1 + dc
            er, ec = rows - 2 + dr, cols - 2 + dc
            if 0 < sr < rows - 1 and 0 < sc < cols - 1:
                maze[sr, sc] = 0
            if 0 < er < rows - 1 and 0 < ec < cols - 1:
                maze[er, ec] = 0

    return maze


def get_preset_mazes() -> dict:
    """Return dictionary of preset mazes with metadata."""
    presets = {}

    # ── Small mazes (10-15) ──
    presets["small_simple"] = {
        "name": "Mały prosty (11x11)",
        "description": "Idealny do testów i nauki algorytmu",
        "generator": "dfs",
        "rows": 11,
        "cols": 11,
        "seed": 42,
        "difficulty": "łatwy",
        "recommended_params": {
            "population_size": 20,
            "max_iterations": 50,
            "z": 0.03,
        },
    }

    presets["small_prim"] = {
        "name": "Mały rozgałęziony (11x11)",
        "description": "Więcej rozgałęzień, trudniejszy dla SMA",
        "generator": "prim",
        "rows": 11,
        "cols": 11,
        "seed": 42,
        "difficulty": "łatwy",
        "recommended_params": {
            "population_size": 25,
            "max_iterations": 80,
            "z": 0.03,
        },
    }

    # ── Medium mazes (21-31) ──
    presets["medium_dfs"] = {
        "name": "Średni DFS (21x21)",
        "description": "Klasyczny labirynt średniej wielkości",
        "generator": "dfs",
        "rows": 21,
        "cols": 21,
        "seed": 123,
        "difficulty": "średni",
        "recommended_params": {
            "population_size": 40,
            "max_iterations": 150,
            "z": 0.03,
        },
    }

    presets["medium_prim"] = {
        "name": "Średni Prim (21x21)",
        "description": "Gęsty labirynt z wieloma ścieżkami",
        "generator": "prim",
        "rows": 21,
        "cols": 21,
        "seed": 456,
        "difficulty": "średni",
        "recommended_params": {
            "population_size": 50,
            "max_iterations": 200,
            "z": 0.03,
        },
    }

    presets["medium_open"] = {
        "name": "Średni otwarty (21x21)",
        "description": "Otwarty labirynt z losowymi ścianami (30%)",
        "generator": "open",
        "rows": 21,
        "cols": 21,
        "seed": 789,
        "wall_density": 0.3,
        "difficulty": "łatwy",
        "recommended_params": {
            "population_size": 30,
            "max_iterations": 100,
            "z": 0.03,
        },
    }

    # ── Large mazes (31-51) ──
    presets["large_dfs"] = {
        "name": "Duży DFS (31x31)",
        "description": "Duży labirynt - prawdziwe wyzwanie",
        "generator": "dfs",
        "rows": 31,
        "cols": 31,
        "seed": 2024,
        "difficulty": "trudny",
        "recommended_params": {
            "population_size": 80,
            "max_iterations": 300,
            "z": 0.03,
        },
    }

    presets["large_prim"] = {
        "name": "Duży Prim (31x31)",
        "description": "Złożony labirynt z wieloma rozgałęzieniami",
        "generator": "prim",
        "rows": 31,
        "cols": 31,
        "seed": 2024,
        "difficulty": "trudny",
        "recommended_params": {
            "population_size": 100,
            "max_iterations": 400,
            "z": 0.03,
        },
    }

    presets["large_open"] = {
        "name": "Duży otwarty (31x31)",
        "description": "Duży labirynt otwarty - wiele dróg",
        "generator": "open",
        "rows": 31,
        "cols": 31,
        "seed": 2024,
        "wall_density": 0.3,
        "difficulty": "średni",
        "recommended_params": {
            "population_size": 60,
            "max_iterations": 200,
            "z": 0.03,
        },
    }

    # ── Extra Large (51+) ──
    presets["xl_dfs"] = {
        "name": "Ekstra duży DFS (51x51)",
        "description": "Ogromny labirynt - test wydajności",
        "generator": "dfs",
        "rows": 51,
        "cols": 51,
        "seed": 9999,
        "difficulty": "ekstremalny",
        "recommended_params": {
            "population_size": 120,
            "max_iterations": 500,
            "z": 0.03,
        },
    }

    presets["xl_open"] = {
        "name": "Ekstra duży otwarty (51x51)",
        "description": "Ogromny otwarty labirynt",
        "generator": "open",
        "rows": 51,
        "cols": 51,
        "seed": 9999,
        "wall_density": 0.3,
        "difficulty": "trudny",
        "recommended_params": {
            "population_size": 100,
            "max_iterations": 400,
            "z": 0.03,
        },
    }

    return presets


def build_maze(preset_key: str) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Build a maze from preset configuration."""
    presets = get_preset_mazes()
    config = presets[preset_key]

    if config["generator"] == "dfs":
        maze = generate_dfs_maze(config["rows"], config["cols"], config.get("seed"))
    elif config["generator"] == "prim":
        maze = generate_prim_maze(config["rows"], config["cols"], config.get("seed"))
    elif config["generator"] == "open":
        maze = generate_open_maze(
            config["rows"], config["cols"],
            config.get("wall_density", 0.3),
            config.get("seed"),
        )
    else:
        maze = generate_dfs_maze(config["rows"], config["cols"])

    rows, cols = maze.shape

    # Find start and end
    start = (1, 1)
    end = (rows - 2, cols - 2)

    # Make sure start and end are walkable
    maze[start[0], start[1]] = 0
    maze[end[0], end[1]] = 0

    return maze, start, end


def build_custom_maze(
    rows: int, cols: int, generator: str = "dfs",
    wall_density: float = 0.3, seed: int = None
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Build a custom maze with given parameters."""
    if generator == "dfs":
        maze = generate_dfs_maze(rows, cols, seed)
    elif generator == "prim":
        maze = generate_prim_maze(rows, cols, seed)
    elif generator == "open":
        maze = generate_open_maze(rows, cols, wall_density, seed)
    else:
        maze = generate_dfs_maze(rows, cols, seed)

    r, c = maze.shape
    start = (1, 1)
    end = (r - 2, c - 2)
    maze[start[0], start[1]] = 0
    maze[end[0], end[1]] = 0

    return maze, start, end
