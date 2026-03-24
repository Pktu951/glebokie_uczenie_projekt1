"""
SMA Maze Solver - Flask Web Application
========================================
Web-based GUI for Slime Mould Algorithm maze pathfinding.
"""

from flask import Flask, render_template, request, jsonify
from flask.json.provider import DefaultJSONProvider
import numpy as np
import json
import time


class NumpyJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that handles numpy types."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

from sma_algorithm import MazeSMA
from classic_algorithms import solve_astar, solve_dijkstra, solve_bfs
from maze_generator import get_preset_mazes, build_maze, build_custom_maze

app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

# Store current maze state
current_state = {}


@app.route("/")
def index():
    """Main page."""
    presets = get_preset_mazes()
    return render_template("index.html", presets=presets)


@app.route("/api/generate_maze", methods=["POST"])
def api_generate_maze():
    """Generate a maze from preset or custom config."""
    data = request.json

    if data.get("preset"):
        maze, start, end = build_maze(data["preset"])
        presets = get_preset_mazes()
        recommended = presets[data["preset"]]["recommended_params"]
    else:
        rows = int(data.get("rows", 21))
        cols = int(data.get("cols", 21))
        generator = data.get("generator", "dfs")
        wall_density = float(data.get("wall_density", 0.3))
        seed = int(data.get("seed")) if data.get("seed") else None

        maze, start, end = build_custom_maze(rows, cols, generator, wall_density, seed)
        recommended = {
            "population_size": max(20, min(rows, cols) * 3),
            "max_iterations": max(50, min(rows, cols) * 10),
            "z": 0.03,
        }

    # Store in state
    current_state["maze"] = maze
    current_state["start"] = start
    current_state["end"] = end

    return jsonify({
        "maze": maze.tolist(),
        "start": list(start),
        "end": list(end),
        "rows": maze.shape[0],
        "cols": maze.shape[1],
        "recommended_params": recommended,
    })


@app.route("/api/solve", methods=["POST"])
def api_solve():
    """Run SMA and classical algorithms on current maze."""
    data = request.json

    maze = np.array(current_state.get("maze"))
    start = tuple(current_state.get("start"))
    end = tuple(current_state.get("end"))

    if maze is None:
        return jsonify({"error": "No maze generated"}), 400

    # SMA parameters
    population_size = int(data.get("population_size", 50))
    max_iterations = int(data.get("max_iterations", 200))
    z_param = float(data.get("z", 0.03))

    results = {}

    # ── Run SMA ──
    sma = MazeSMA(
        maze=maze,
        start=start,
        end=end,
        population_size=population_size,
        max_iterations=max_iterations,
        z=z_param,
    )
    sma_result = sma.solve()

    results["sma"] = {
        "name": "Slime Mould Algorithm",
        "path": [list(p) for p in sma_result.best_path] if sma_result.best_path else [],
        "path_length": len(sma_result.best_path) if sma_result.best_path else 0,
        "fitness": sma_result.best_fitness,
        "execution_time": round(sma_result.execution_time, 4),
        "iterations_used": sma_result.iterations_used,
        "convergence": sma_result.convergence_history,
        "found_end": list(sma_result.best_path[-1]) == list(end) if sma_result.best_path else False,
        "params": sma_result.params_used,
    }

    # ── Run A* ──
    astar_result = solve_astar(maze, start, end)
    results["astar"] = {
        "name": "A*",
        "path": [list(p) for p in astar_result.path],
        "path_length": astar_result.path_length,
        "nodes_explored": astar_result.nodes_explored,
        "execution_time": round(astar_result.execution_time, 6),
        "explored_cells": [list(c) for c in astar_result.explored_cells[:500]],
    }

    # ── Run Dijkstra ──
    dijkstra_result = solve_dijkstra(maze, start, end)
    results["dijkstra"] = {
        "name": "Dijkstra",
        "path": [list(p) for p in dijkstra_result.path],
        "path_length": dijkstra_result.path_length,
        "nodes_explored": dijkstra_result.nodes_explored,
        "execution_time": round(dijkstra_result.execution_time, 6),
        "explored_cells": [list(c) for c in dijkstra_result.explored_cells[:500]],
    }

    # ── Run BFS ──
    bfs_result = solve_bfs(maze, start, end)
    results["bfs"] = {
        "name": "BFS",
        "path": [list(p) for p in bfs_result.path],
        "path_length": bfs_result.path_length,
        "nodes_explored": bfs_result.nodes_explored,
        "execution_time": round(bfs_result.execution_time, 6),
        "explored_cells": [list(c) for c in bfs_result.explored_cells[:500]],
    }

    # ── Comparison summary ──
    optimal_length = astar_result.path_length
    sma_length = len(sma_result.best_path) if sma_result.best_path else 0
    sma_reached = list(sma_result.best_path[-1]) == list(end) if sma_result.best_path else False

    results["comparison"] = {
        "optimal_path_length": optimal_length,
        "sma_path_length": sma_length,
        "sma_reached_end": sma_reached,
        "sma_optimality_ratio": round(sma_length / optimal_length, 2) if optimal_length > 0 and sma_reached else None,
        "speed_comparison": {
            "sma": round(sma_result.execution_time, 4),
            "astar": round(astar_result.execution_time, 6),
            "dijkstra": round(dijkstra_result.execution_time, 6),
            "bfs": round(bfs_result.execution_time, 6),
        },
    }

    return jsonify(results)


@app.route("/api/presets")
def api_presets():
    """Return available maze presets."""
    return jsonify(get_preset_mazes())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
