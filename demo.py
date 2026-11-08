from planner.grid import Grid
from planner.dijkstra import dijkstra
from planner.viz import show

ROWS, COLS = 18, 25
shelves = set()
for top in (3, 8, 13):                      # three shelf rows
    for left in (3, 9, 15):                 # three shelf columns
        for r in range(top, top + 2):
            for c in range(left, left + 4):
                shelves.add((r, c))

grid = Grid(ROWS, COLS, shelves)
start, goal = (1, 1), (16, 23)

result = dijkstra(grid, start, goal)
print("found:", result.found, "| cost:", result.cost,
      "| explored:", len(result.explored), "| time_ms:", round(result.time_ms, 3))
show(grid, start, goal, result, "Dijkstra")