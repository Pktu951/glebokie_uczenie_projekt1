"""
Classical Pathfinding Algorithms for Comparison
================================================
A*, Dijkstra, BFS - used to benchmark SMA performance.
"""

import heapq
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class ClassicResult:
    """Result container for classical algorithms."""
    algorithm_name: str
    path: List[Tuple[int, int]]
    path_length: int
    nodes_explored: int
    execution_time: float
    explored_cells: List[Tuple[int, int]]  # For visualization


def _reconstruct_path(
    came_from: Dict, current: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Reconstruct path from came_from dictionary."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _get_neighbors(
    maze: np.ndarray, pos: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Get valid neighbors (4-directional)."""
    rows, cols = maze.shape
    r, c = pos
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0:
            neighbors.append((nr, nc))
    return neighbors


def solve_astar(
    maze: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> ClassicResult:
    """
    A* Algorithm - optimal pathfinding with heuristic.
    
    f(n) = g(n) + h(n)
    - g(n): cost from start to n
    - h(n): Manhattan distance heuristic to end
    """
    start_time = time.time()

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    explored = []
    nodes_explored = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_explored += 1
        explored.append(current)

        if current == end:
            path = _reconstruct_path(came_from, current)
            return ClassicResult(
                algorithm_name="A*",
                path=path,
                path_length=len(path),
                nodes_explored=nodes_explored,
                execution_time=time.time() - start_time,
                explored_cells=explored,
            )

        for neighbor in _get_neighbors(maze, current):
            tentative_g = g_score[current] + 1

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                f = tentative_g + h
                heapq.heappush(open_set, (f, neighbor))

    return ClassicResult(
        algorithm_name="A*",
        path=[],
        path_length=0,
        nodes_explored=nodes_explored,
        execution_time=time.time() - start_time,
        explored_cells=explored,
    )


def solve_dijkstra(
    maze: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> ClassicResult:
    """
    Dijkstra's Algorithm - optimal pathfinding without heuristic.
    Guarantees shortest path.
    """
    start_time = time.time()

    open_set = [(0, start)]
    came_from = {}
    dist = {start: 0}
    explored = []
    nodes_explored = 0

    while open_set:
        d, current = heapq.heappop(open_set)
        nodes_explored += 1
        explored.append(current)

        if current == end:
            path = _reconstruct_path(came_from, current)
            return ClassicResult(
                algorithm_name="Dijkstra",
                path=path,
                path_length=len(path),
                nodes_explored=nodes_explored,
                execution_time=time.time() - start_time,
                explored_cells=explored,
            )

        if d > dist.get(current, float('inf')):
            continue

        for neighbor in _get_neighbors(maze, current):
            new_dist = dist[current] + 1
            if new_dist < dist.get(neighbor, float('inf')):
                dist[neighbor] = new_dist
                came_from[neighbor] = current
                heapq.heappush(open_set, (new_dist, neighbor))

    return ClassicResult(
        algorithm_name="Dijkstra",
        path=[],
        path_length=0,
        nodes_explored=nodes_explored,
        execution_time=time.time() - start_time,
        explored_cells=explored,
    )


def solve_bfs(
    maze: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> ClassicResult:
    """
    Breadth-First Search - guarantees shortest path in unweighted graph.
    """
    start_time = time.time()

    queue = deque([start])
    came_from = {}
    visited = {start}
    explored = []
    nodes_explored = 0

    while queue:
        current = queue.popleft()
        nodes_explored += 1
        explored.append(current)

        if current == end:
            path = _reconstruct_path(came_from, current)
            return ClassicResult(
                algorithm_name="BFS",
                path=path,
                path_length=len(path),
                nodes_explored=nodes_explored,
                execution_time=time.time() - start_time,
                explored_cells=explored,
            )

        for neighbor in _get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return ClassicResult(
        algorithm_name="BFS",
        path=[],
        path_length=0,
        nodes_explored=nodes_explored,
        execution_time=time.time() - start_time,
        explored_cells=explored,
    )
