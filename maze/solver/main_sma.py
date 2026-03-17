from sma import MazeSolverSMA, MAZE_LARGE
from visualizer_sma import MazeVisualizer


# Rozwiąż
solver = MazeSolverSMA(maze=MAZE_LARGE, params={'num_agents': 20})
best_path, all_sol, freq_map, stats = solver.solve()

# Wyświetl(Jupyter, jesli do terminala albo zapis do pliku to trzeba zmienic)
viz = MazeVisualizer(solver.maze, solver.start, solver.goal)
viz.visualize_solution(best_path, freq_map, solver.global_trail)