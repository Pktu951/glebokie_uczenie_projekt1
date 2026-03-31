"""
Slime Mould Algorithm (SMA) for Maze Pathfinding
==================================================
Based on: Li et al. (2020) - "Slime mould algorithm: A new method for stochastic optimization"
Adapted for discrete maze pathfinding problem.

The SMA simulates the oscillatory behavior of slime mould (Physarum polycephalum)
during foraging. Each "agent" represents a candidate path through the maze.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import random


@dataclass
class SMAResult:
    """Result container for SMA execution."""
    best_path: List[Tuple[int, int]]
    best_fitness: float
    convergence_history: List[float]
    all_paths_explored: List[List[Tuple[int, int]]]
    execution_time: float
    iterations_used: int
    params_used: Dict


class MazeSMA:
    """
    Slime Mould Algorithm adapted for maze pathfinding.
    
    Mathematical Model (from the paper):
    ─────────────────────────────────────
    Position update equation:
        X(t+1) = { rand*(UB-LB) + LB,                    if rand < z
                  { Xb(t) + vb * (W * XA(t) - XB(t)),    if r < p
                  { vc * X(t),                             if r >= p
    
    Where:
        - z = 0.03 (exploration probability)
        - p = tanh|S(i) - DF| (fitness-based probability)
        - a = arctanh(1 - t/max_t) (adaptive boundary)
        - vb ∈ [-a, a] (oscillation parameter)
        - vc linearly decreases from 1 to 0
        - W = weight based on fitness ranking (Eq. 4)
    
    Adaptation for maze:
        - Each agent is a path (sequence of cells) from start to end
        - Fitness = path length + penalties for invalid moves
        - Position update = path modification via crossover/mutation
    """

    def __init__(
        self,
        maze: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        population_size: int = 50,
        max_iterations: int = 200,
        z: float = 0.03,
    ):
        self.maze = maze
        self.start = start
        self.end = end
        self.rows, self.cols = maze.shape
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.z = z  # Exploration probability parameter

        # Precompute valid neighbors for each cell
        self._neighbor_cache = {}
        self._precompute_neighbors()

    def _precompute_neighbors(self):
        """Precompute valid neighbors for each walkable cell."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze[r, c] == 0:  # Walkable
                    neighbors = []
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.maze[nr, nc] == 0:
                            neighbors.append((nr, nc))
                    self._neighbor_cache[(r, c)] = neighbors

    def _generate_random_path(self) -> List[Tuple[int, int]]:
        """
        Generate a random valid path from start to end using biased random walk.
        Uses a greedy-random hybrid approach to ensure path validity.
        """
        path = [self.start]
        visited = {self.start}
        current = self.start
        max_steps = self.rows * self.cols * 2

        for _ in range(max_steps):
            if current == self.end:
                return path

            neighbors = self._neighbor_cache.get(current, [])
            unvisited = [n for n in neighbors if n not in visited]

            if not unvisited:
                # Backtrack
                if len(path) > 1:
                    path.pop()
                    current = path[-1]
                    continue
                else:
                    break

            # Bias towards end point (greedy component)
            distances = []
            for n in unvisited:
                dist = abs(n[0] - self.end[0]) + abs(n[1] - self.end[1])
                distances.append(dist)

            # Softmax-like probability with randomness
            min_dist = min(distances)
            weights = [np.exp(-0.5 * (d - min_dist)) for d in distances]
            total = sum(weights)
            probs = [w / total for w in weights]

            chosen = random.choices(unvisited, weights=probs, k=1)[0]
            path.append(chosen)
            visited.add(chosen)
            current = chosen

        return path

    def _fitness(self, path: List[Tuple[int, int]]) -> float:
        """
        Evaluate fitness of a path. Lower is better.
        
        Fitness = path_length + penalty_if_not_reaching_end
        """
        if not path:
            return float('inf')

        # Heavy penalty if path doesn't reach end
        last = path[-1]
        distance_to_end = abs(last[0] - self.end[0]) + abs(last[1] - self.end[1])

        if distance_to_end == 0:
            return len(path)  # Pure path length
        else:
            return len(path) + distance_to_end * 100

    def _calculate_weight(
        self, fitness_values: np.ndarray, smell_index: np.ndarray
    ) -> np.ndarray:
        """
        Calculate slime mould weight W (Equation 4 from paper).
        
        W(SmellIndex(i)) = { 1 + r*log((bF-S(i))/(bF-wF) + 1),  if in first half
                           { 1 - r*log((bF-S(i))/(bF-wF) + 1),  otherwise
        """
        n = len(fitness_values)
        W = np.zeros(n)
        bF = fitness_values[smell_index[0]]      # Best fitness
        wF = fitness_values[smell_index[-1]]      # Worst fitness

        for i in range(n):
            idx = smell_index[i]
            r = np.random.random()

            if bF == wF:
                W[idx] = 1.0
            else:
                ratio = (bF - fitness_values[idx]) / (bF - wF + 1e-10) + 1
                log_val = np.log(max(ratio, 1e-10))

                if i < n // 2:  # First half (better solutions)
                    W[idx] = 1 + r * log_val
                else:  # Second half (worse solutions)
                    W[idx] = 1 - r * log_val

        return W

    def _crossover_paths(
        self, path1: List[Tuple[int, int]], path2: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Crossover two paths at common cells.
        This implements the SMA position update: Xb + vb*(W*XA - XB)
        adapted for discrete paths.
        """
        # Find common cells
        set1 = set(path1)
        set2 = set(path2)
        common = set1 & set2

        if len(common) <= 1:
            # No useful crossover point, return shorter path with mutation
            return path1 if len(path1) <= len(path2) else path2

        common_list = [c for c in path1 if c in common]
        if len(common_list) < 2:
            return path1

        # Pick a random crossover point
        cross_cell = random.choice(common_list[1:-1]) if len(common_list) > 2 else common_list[0]

        # Build hybrid path
        idx1 = path1.index(cross_cell)
        idx2 = path2.index(cross_cell)

        # Take first part from path1, second from path2
        new_path = path1[: idx1 + 1] + path2[idx2 + 1:]

        # Validate connectivity
        if self._is_valid_path(new_path):
            return new_path
        return path1 if self._fitness(path1) <= self._fitness(path2) else path2

    def _mutate_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Mutate a path by re-routing a random segment.
        Implements exploration component of SMA.
        """
        if len(path) < 4:
            return path

        # Pick two random points on the path
        i = random.randint(1, len(path) - 3)
        j = random.randint(i + 1, len(path) - 1)

        # Try to find a new sub-path between path[i] and path[j]
        sub_path = self._find_sub_path(path[i], path[j])

        if sub_path:
            new_path = path[:i] + sub_path + path[j + 1:]
            if self._is_valid_path(new_path):
                return new_path

        return path

    def _find_sub_path(
        self, start: Tuple[int, int], end: Tuple[int, int], max_steps: int = 100
    ) -> Optional[List[Tuple[int, int]]]:
        """Find a sub-path using biased random walk."""
        path = [start]
        visited = {start}
        current = start

        for _ in range(max_steps):
            if current == end:
                return path

            neighbors = self._neighbor_cache.get(current, [])
            unvisited = [n for n in neighbors if n not in visited]

            if not unvisited:
                return None

            # Bias towards end
            distances = [abs(n[0] - end[0]) + abs(n[1] - end[1]) for n in unvisited]
            min_d = min(distances)
            weights = [np.exp(-(d - min_d)) for d in distances]
            total = sum(weights)
            probs = [w / total for w in weights]

            chosen = random.choices(unvisited, weights=probs, k=1)[0]
            path.append(chosen)
            visited.add(chosen)
            current = chosen

        return None

    def _is_valid_path(self, path: List[Tuple[int, int]]) -> bool:
        """Check if path is valid (all steps are to adjacent walkable cells)."""
        if not path:
            return False
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            if abs(r1 - r2) + abs(c1 - c2) != 1:
                return False
            if self.maze[r2, c2] != 0:
                return False
        return True

    def _local_optimize(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Remove loops from path (simple local optimization)."""
        if not path:
            return path

        seen = {}
        optimized = []

        for i, cell in enumerate(path):
            if cell in seen:
                # Remove the loop
                optimized = optimized[: seen[cell]]
            seen[cell] = len(optimized)
            optimized.append(cell)

        return optimized

    def solve(self, callback=None) -> SMAResult:
        """
        Run the Slime Mould Algorithm to find the shortest path.
        
        SMA Main Loop:
        1. Initialize population of random paths
        2. For each iteration:
           a. Calculate fitness for all agents
           b. Sort by fitness (SmellIndex)
           c. Update weights W
           d. Calculate adaptive parameters (a, vb, vc, p)
           e. Update positions using Equation (1)
        3. Return best path found
        
        Parameters callback: Optional function called each iteration with
                            (iteration, best_fitness, best_path) for visualization.
        """
        start_time = time.time()
        
        params_used = {
            "population_size": self.population_size,
            "max_iterations": self.max_iterations,
            "z": self.z,
            "maze_size": f"{self.rows}x{self.cols}",
        }

        # ── Step 1: Initialize population ──
        population = []
        for _ in range(self.population_size):
            path = self._generate_random_path()
            path = self._local_optimize(path)
            population.append(path)

        convergence_history = []
        all_explored = []
        best_ever_path = None
        best_ever_fitness = float('inf')

        # ── Step 2: Main iteration loop ──
        for t in range(self.max_iterations):
            # (a) Calculate fitness
            fitness_values = np.array([self._fitness(p) for p in population])

            # (b) Sort by fitness → SmellIndex (Eq. 5)
            smell_index = np.argsort(fitness_values)

            # Track best
            current_best_idx = smell_index[0]
            current_best_fitness = fitness_values[current_best_idx]

            if current_best_fitness < best_ever_fitness:
                best_ever_fitness = current_best_fitness
                best_ever_path = population[current_best_idx].copy()

            convergence_history.append(best_ever_fitness)

            # (c) Calculate weight W (Eq. 4)
            W = self._calculate_weight(fitness_values, smell_index)

            # (d) Calculate adaptive parameters
            # a = arctanh(1 - t/max_t) (Eq. 3)
            ratio = 1 - (t + 1) / self.max_iterations
            a = np.arctanh(np.clip(ratio, -0.999, 0.999))

            # vc linearly decreases from 1 to 0
            vc = 1 - t / self.max_iterations

            # (e) Update positions (Eq. 1)
            new_population = []
            for i in range(self.population_size):
                rand_val = np.random.random()
                r = np.random.random()

                # p = tanh|S(i) - DF| (Eq. 2)
                p = np.tanh(abs(fitness_values[i] - best_ever_fitness))

                if rand_val < self.z:
                    # ── Equation 1, case 1: Random exploration ──
                    # X(t+1) = rand*(UB-LB) + LB
                    new_path = self._generate_random_path()
                    new_path = self._local_optimize(new_path)

                elif r < p:
                    # ── Equation 1, case 2: Exploitation towards best ──
                    # X(t+1) = Xb + vb*(W*XA - XB)
                    vb = random.uniform(-a, a)

                    # XA, XB = two random agents
                    idx_A = random.randint(0, self.population_size - 1)
                    idx_B = random.randint(0, self.population_size - 1)

                    # Crossover best path with weighted combination of XA, XB
                    if W[i] > 1.0:
                        # Higher weight → exploit best path more
                        new_path = self._crossover_paths(
                            best_ever_path, population[idx_A]
                        )
                    else:
                        new_path = self._crossover_paths(
                            population[idx_A], population[idx_B]
                        )

                    # Apply vb-influenced mutation
                    if abs(vb) > 0.5:
                        new_path = self._mutate_path(new_path)

                    new_path = self._local_optimize(new_path)

                else:
                    # ── Equation 1, case 3: Contraction ──
                    # X(t+1) = vc * X(t)
                    # In maze context: slightly modify current path
                    if vc > 0.5:
                        new_path = self._mutate_path(population[i])
                    else:
                        # Late iterations: fine-tune with small mutations
                        new_path = self._local_optimize(population[i])

                    new_path = self._local_optimize(new_path)

                new_population.append(new_path)

            population = new_population
            all_explored.extend([p.copy() for p in population[:3]])

            # Callback for visualization
            if callback:
                callback(t, best_ever_fitness, best_ever_path)

            # Early stopping if optimal-looking solution found
            if best_ever_path and best_ever_path[-1] == self.end:
                if t > 20 and len(set(convergence_history[-20:])) == 1:
                    break

        execution_time = time.time() - start_time

        return SMAResult(
            best_path=best_ever_path if best_ever_path else [],
            best_fitness=best_ever_fitness,
            convergence_history=convergence_history,
            all_paths_explored=all_explored[-100:],  # Keep last 100
            execution_time=execution_time,
            iterations_used=len(convergence_history),
            params_used=params_used,
        )
