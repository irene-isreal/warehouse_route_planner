from planner.grid import Grid
from planner.dijkstra import dijkstra
from planner.astar import astar
from planner.viz import compare

ROWS, COLS = 18, 25
shelves = set()
for top in (3, 8, 13):
    for left in (3, 9, 15):
        for r in range(top, top + 2):
            for c in range(left, left + 4):
                shelves.add((r, c))

grid = Grid(ROWS, COLS, shelves)
start, goal = (1, 1), (16, 23)

results = {"Dijkstra": dijkstra(grid, start, goal), "A*": astar(grid, start, goal)}
for name, res in results.items():
    print(f"{name:9} cost={res.cost}  explored={len(res.explored)}  time_ms={res.time_ms:.3f}")

compare(grid, start, goal, results)