import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any


# --- DOMYSLNE LABIRYNTY ---
MAZE_SMALL = np.array([
    [1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,1,0,0,0,0,1],
    [1,1,1,0,1,1,0,1,1,0,1],
    [1,0,0,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1]
])

MAZE_LARGE = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,0,1,0,1,1,0,1,0,1,1,0,1],
    [1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,1],
    [1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])

# --- DOMYSLNE PARAMETRY ---
DEFAULT_PARAMS = {
    'num_agents': 15,           # Liczba agentów
    'max_steps': 80,            # Max kroków per agent per iteracja
    'trail_decay': 0.99,        # Zanikanie śladu per iteracja (0.95-0.99)
    'num_desired_paths': 10,    # Liczba szukanych rozwiązań
    'max_iterations': 100,      # Max liczba iteracji
    'stuck_threshold': 5,       # Ile kroków bez poprawy przed głębokim backtrackingiem
    'backtrack_depth': 3        # Ile kroków wstecz przy utknięciu
}


class MazeSolverSMA:
    """Solver labiryntu za pomoca Swarm Multi-Agent (SMA) z inteligentnym backtrackingiem."""
    
    def __init__(self, 
                 maze: Optional[np.ndarray] = None,
                 start: Optional[Tuple[int, int]] = None,
                 goal: Optional[Tuple[int, int]] = None,
                 params: Optional[Dict[str, Any]] = None):
        """
        Inicjalizacja solvera.
        
        Args:
            maze: Tablica labiryntu (1=ściana, 0=przejście). Default: MAZE_LARGE
            start: Pozycja startowa (row, col). Default: (1, 1)
            goal: Pozycja celu (row, col). Default: ostatnie wolne pole
            params: Słownik parametrów. Default: DEFAULT_PARAMS
                   Dostępne klucze:
                   - num_agents: liczba agentów (domyślnie 15)
                   - max_steps: kroków per agent (domyślnie 80)
                   - trail_decay: zanikanie śladu 0.0-1.0 (domyślnie 0.99)
                   - num_desired_paths: liczba szukanych rozwiązań (domyślnie 10)
                   - max_iterations: max iteracji (domyślnie 100)
                   - stuck_threshold: kroki bez poprawy (domyślnie 5)
                   - backtrack_depth: głębokość backtrackingu (domyślnie 3)
        """
        self.maze = maze if maze is not None else MAZE_LARGE.copy()
        self.start = start if start is not None else (1, 1)
        self.goal = goal if goal is not None else (self.maze.shape[0] - 2, self.maze.shape[1] - 2)
        
        # Merge parametrów
        self.params = DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)
        
        # Walidacja parametrów
        self.params['trail_decay'] = np.clip(self.params['trail_decay'], 0.0, 1.0)
        self.params['stuck_threshold'] = max(1, self.params['stuck_threshold'])
        self.params['backtrack_depth'] = max(1, self.params['backtrack_depth'])
        
        self.global_trail = np.zeros_like(self.maze, dtype=float)
        self.solutions: List[List[Tuple[int, int]]] = []
        self.freq_map = np.zeros_like(self.maze, dtype=float)
        
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Zwraca dostępne sąsiednie pola w labiryncie."""
        r, c = pos
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        neighbors = []
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.maze.shape[0] and 
                0 <= nc < self.maze.shape[1] and 
                self.maze[nr, nc] == 0):
                neighbors.append((nr, nc))
        return neighbors
    
    def manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Odległość Manhattan między punktami."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _smart_backtrack(self, agent: Dict, steps_stuck: int) -> None:
        """
        Inteligentny backtracking - cofanie się z głębokich ślepych ścieżek.
        
        Args:
            agent: Słownik agenta
            steps_stuck: Liczba kroków bez postępu
        """
        backtrack_depth = self.params['backtrack_depth']
        
        # Cofnij się o N kroków
        depth = min(backtrack_depth, len(agent["history"]) - 1)
        for _ in range(depth):
            if len(agent["history"]) > 1:
                agent["history"].pop()
        
        agent["pos"] = agent["history"][-1]
        agent["last_distance"] = self.manhattan(agent["pos"], self.goal)
        agent["stuck_count"] = 0
    
    def extract_best_path(self) -> List[Tuple[int, int]]:
        """
        Zwraca najkrótszą znalezioną ścieżkę.
        Jeśli brak rozwiązań - zwraca tylko start.
        """
        if len(self.solutions) == 0:
            return [self.start]
        
        # Zwróć najkrótszą ścieżkę ze znalezionych
        return min(self.solutions, key=len)
    
    def solve(self) -> Tuple[List[Tuple[int, int]], List[List[Tuple[int, int]]], np.ndarray, Dict[str, Any]]:
        """
        Rozwiąż labirynt metodą SMA z inteligentnym backtrackingiem.
        
        Returns:
            best_path: Dominująca ścieżka
            all_solutions: Wszystkie znalezione rozwiązania
            freq_map: Mapa częstotliwości
            stats: Słownik statystyk
        """
        num_agents = self.params['num_agents']
        max_steps = self.params['max_steps']
        num_desired_paths = self.params['num_desired_paths']
        max_iterations = self.params['max_iterations']
        trail_decay = self.params['trail_decay']
        stuck_threshold = self.params['stuck_threshold']
        
        iteration = 0
        
        while len(self.solutions) < num_desired_paths and iteration < max_iterations:
            iteration += 1
            agents = []
            
            # Inicjalizacja agentów
            for _ in range(num_agents):
                agents.append({
                    "pos": self.start,
                    "history": [self.start],
                    "last_distance": self.manhattan(self.start, self.goal),
                    "stuck_count": 0,
                    "solved": False
                })
            
            # Symulacja iteracji
            for step in range(max_steps):
                for a in agents:
                    if a["solved"]:
                        continue
                    
                    neighbors = self.get_neighbors(a["pos"])
                    
                    # Nie wracaj do ostatniej pozycji
                    if len(a["history"]) >= 2:
                        last = a["history"][-2]
                        neighbors = [n for n in neighbors if n != last]
                    
                    if not neighbors:
                        # Brak sąsiadów - backtrack
                        if len(a["history"]) > 1:
                            a["pos"] = a["history"][-2]
                            a["history"].pop()
                            a["stuck_count"] += 1
                        continue
                    
                    # Wybór ruchu: trail + heurystyka + losowość
                    scores = []
                    for n in neighbors:
                        trail = self.global_trail[n]
                        heuristic = 1.0 / (1.0 + self.manhattan(n, self.goal))
                        score = trail + heuristic + random.random() * 0.1
                        scores.append(score)
                    
                    next_pos = neighbors[np.argmax(scores)]
                    a["pos"] = next_pos
                    a["history"].append(next_pos)
                    
                    # Sprawdź postęp w stosunku do celu
                    new_distance = self.manhattan(next_pos, self.goal)
                    if new_distance < a["last_distance"]:
                        a["stuck_count"] = 0  # Reset licznika utknięcia
                        a["last_distance"] = new_distance
                    else:
                        a["stuck_count"] += 1
                    
                    # Inteligentny backtracking - jeśli utknął
                    if a["stuck_count"] >= stuck_threshold and len(a["history"]) > stuck_threshold + 1:
                        self._smart_backtrack(a, a["stuck_count"])
                    
                    # Sprawdzenie dotarcia do celu
                    if next_pos == self.goal and not a["solved"]:
                        a["solved"] = True
                        self.solutions.append(a["history"][:])
                        # Wzmocnienie śladu dla sukcesu
                        for pos in a["history"]:
                            self.global_trail[pos] += 5.0
                
                # Zanik śladu
                self.global_trail *= trail_decay
        
        best_path = self.extract_best_path()
        
        # Buduj freq_map na końcu do wizualizacji
        self.freq_map = np.zeros_like(self.maze, dtype=float)
        for solution in self.solutions:
            for pos in solution:
                self.freq_map[pos] += 1.0
        
        stats = {
            'iterations': iteration,
            'solutions_found': len(self.solutions),
            'best_path_length': len(best_path),
            'all_paths_lengths': [len(s) for s in self.solutions],
            'max_trail_value': float(self.global_trail.max()),
            'params_used': self.params.copy()
        }
        
        return best_path, self.solutions, self.freq_map, stats


if __name__ == "__main__":
    # Test 1: Domyślne parametry
    print("📍 Test 1: Domyślne parametry")
    solver = MazeSolverSMA()
    best_path, all_sol, freq_map, stats = solver.solve()
    print(f"   Znalezionych: {stats['solutions_found']}")
    if stats['solutions_found'] > 0:
        print(f"   Dominująca ścieżka: {stats['best_path_length']} kroków")
    print(f"   Iteracje: {stats['iterations']}\n")
    
    # Test 2: Custom parametry - agresywne szukanie
    print("📍 Test 2: Agresywne szukanie (więcej agentów)")
    custom_params = {
        'num_agents': 30,
        'max_steps': 150,
        'num_desired_paths': 8,
        'max_iterations': 60,
        'trail_decay': 0.95,
        'stuck_threshold': 4,
        'backtrack_depth': 4
    }
    solver2 = MazeSolverSMA(params=custom_params)
    best_path2, all_sol2, freq_map2, stats2 = solver2.solve()
    print(f"   Znalezionych: {stats2['solutions_found']}")
    print(f"   Dominująca ścieżka: {stats2['best_path_length']} kroków")
    print(f"   Iteracje: {stats2['iterations']}\n")
    
    # Test 3: Custom labirynt + parametry
    print("📍 Test 3: Mały labirynt ze zmienionymi parametrami")
    custom_params3 = {
        'num_agents': 20,
        'max_steps': 100,
        'num_desired_paths': 5,
        'max_iterations': 50,
        'trail_decay': 0.97,
        'stuck_threshold': 3,
        'backtrack_depth': 2
    }
    solver3 = MazeSolverSMA(
        maze=MAZE_SMALL,
        start=(1, 1),
        goal=(5, 9),
        params=custom_params3
    )
    best_path3, all_sol3, freq_map3, stats3 = solver3.solve()
    print(f"   Znalezionych: {stats3['solutions_found']}")
    print(f"   Dominująca ścieżka: {stats3['best_path_length']} kroków")
    if best_path3:
        print(f"   Ścieżka: {' → '.join([str(p) for p in best_path3])}\n")
    
    # Test 4: Szybkie szukanie (mniej zasobów)
    print("📍 Test 4: Szybkie szukanie (mniej agentów)")
    quick_params = {
        'num_agents': 8,
        'max_steps': 60,
        'num_desired_paths': 3,
        'max_iterations': 25,
        'trail_decay': 0.99,
        'stuck_threshold': 5,
        'backtrack_depth': 2
    }
    solver4 = MazeSolverSMA(maze=MAZE_SMALL, params=quick_params)
    best_path4, all_sol4, freq_map4, stats4 = solver4.solve()
    print(f"   Znalezionych: {stats4['solutions_found']}")
    print(f"   Dominująca ścieżka: {stats4['best_path_length']} kroków")
    print(f"   Iteracje: {stats4['iterations']}\n")
    
    print("✅ Testy zakończone!")