import random

from planner.grid import Grid
from planner.dijkstra import dijkstra
from planner.astar import astar


def test_astar_simple_map():
    grid, s, g = Grid.from_ascii([
        "S..#",
        ".#.#",
        ".#..",
        "...G",
    ])
    result = astar(grid, s, g)
    assert result.found and result.cost == 6


def test_astar_no_path():
    grid, s, g = Grid.from_ascii(["S#G"])
    assert not astar(grid, s, g).found


def test_astar_matches_dijkstra_on_random_maps():
    random.seed(0)
    start, goal = (0, 0), (17, 24)
    for _ in range(200):
        blocked = {(random.randrange(18), random.randrange(25)) for _ in range(90)}
        blocked.discard(start)
        blocked.discard(goal)
        grid = Grid(18, 25, blocked)
        a, d = astar(grid, start, goal), dijkstra(grid, start, goal)
        assert a.found == d.found
        if a.found:
            assert a.cost == d.cost
        assert len(a.explored) <= len(d.explored) + 25  # A* should never be much worse