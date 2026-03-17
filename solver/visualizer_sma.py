"""
maze_visualizer.py - Uproszczona wersja
Tylko ostateczna ścieżka
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class MazeVisualizer:
    """Wizualizacja rozwiązania labiryntu."""
    
    def __init__(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        self.maze = maze
        self.start = start
        self.goal = goal
    
    def visualize_solution(self, best_path: List[Tuple[int, int]], 
                          freq_map: np.ndarray, global_trail: np.ndarray,
                          title: str = "Rozwiązanie labiryntu"):
        """
        Wyświetl rozwiązanie w 3 podwykresy.
        
        Args:
            best_path: Dominująca ścieżka
            freq_map: Mapa częstotliwości
            global_trail: Globalna mapa śladu
            title: Tytuł
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # --- Subplot 1: Mapa częstotliwości + dominująca ścieżka ---
        ax1 = axes[0]
        ax1.imshow(self.maze, cmap='binary', origin='upper', alpha=0.3)
        
        freq_masked = np.ma.masked_where(self.maze == 1, freq_map)
        im1 = ax1.imshow(freq_masked, cmap='YlOrRd', origin='upper', alpha=0.7)
        
        if len(best_path) > 0:
            rows, cols = zip(*best_path)
            ax1.plot(cols, rows, 'b-', linewidth=3, label='Ścieżka')
        
        ax1.scatter(self.start[1], self.start[0], c='green', s=200, 
                   marker='o', edgecolors='black', linewidths=2, label='START', zorder=5)
        ax1.scatter(self.goal[1], self.goal[0], c='red', s=200, 
                   marker='X', edgecolors='black', linewidths=2, label='GOAL', zorder=5)
        
        ax1.set_title('Mapa częstotliwości + Ścieżka', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.2)
        plt.colorbar(im1, ax=ax1, label='Liczba wizyt')
        
        # --- Subplot 2: Globalna mapa śladu ---
        ax2 = axes[1]
        ax2.imshow(self.maze, cmap='binary', origin='upper', alpha=0.3)
        
        trail_masked = np.ma.masked_where(self.maze == 1, global_trail)
        im2 = ax2.imshow(trail_masked, cmap='hot', origin='upper', alpha=0.7, vmin=0, vmax=50)
        
        ax2.scatter(self.start[1], self.start[0], c='green', s=200, 
                   marker='o', edgecolors='black', linewidths=2, label='START', zorder=5)
        ax2.scatter(self.goal[1], self.goal[0], c='red', s=200, 
                   marker='X', edgecolors='black', linewidths=2, label='GOAL', zorder=5)
        
        ax2.set_title('Mapa śladu feromonowego', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.2)
        plt.colorbar(im2, ax=ax2, label='Intensywność śladu')
        
        # --- Subplot 3: Ścieżka ze strzałkami ---
        ax3 = axes[2]
        ax3.imshow(self.maze, cmap='binary', origin='upper', alpha=0.5)
        
        if len(best_path) > 0:
            rows, cols = zip(*best_path)
            ax3.plot(cols, rows, 'b-', linewidth=3, label='Ścieżka')
            
            # Strzałki
            step = max(1, len(best_path) // 10)
            for i in range(0, len(best_path) - step, step):
                y1, x1 = best_path[i]
                y2, x2 = best_path[i + step]
                ax3.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.4, head_length=0.3,
                         fc='blue', ec='blue', alpha=0.6, zorder=3)
        
        ax3.scatter(self.start[1], self.start[0], c='green', s=200, 
                   marker='o', edgecolors='black', linewidths=2, label='START', zorder=5)
        ax3.scatter(self.goal[1], self.goal[0], c='red', s=200, 
                   marker='X', edgecolors='black', linewidths=2, label='GOAL', zorder=5)
        
        ax3.set_title(f'Ścieżka ({len(best_path)} kroków)', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.2)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from maze_solver_sma_improved import MazeSolverSMA, MAZE_SMALL
    
    solver = MazeSolverSMA(
        maze=MAZE_SMALL,
        start=(1, 1),
        goal=(5, 9),
        params={'num_agents': 15, 'max_steps': 100, 'num_desired_paths': 5}
    )
    
    best_path, _, freq_map, stats = solver.solve()
    
    viz = MazeVisualizer(solver.maze, solver.start, solver.goal)
    viz.visualize_solution(best_path, freq_map, solver.global_trail)