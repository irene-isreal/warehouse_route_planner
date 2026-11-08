import statistics
import tracemalloc

from .astar import astar
from .dijkstra import dijkstra

ALGORITHMS = {"Dijkstra": dijkstra, "A*": astar}


def measure(algo, grid, start, goal, repeats=20):
    """Run `algo` several times for timing, once more under tracemalloc for memory."""
    times = []
    result = None
    for _ in range(repeats):
        result = algo(grid, start, goal)
        times.append(result.time_ms)

    tracemalloc.start()
    algo(grid, start, goal)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "found": result.found,
        "cost": result.cost,
        "explored": len(result.explored),
        "time_ms": statistics.median(times),
        "peak_kb": peak / 1024,
    }