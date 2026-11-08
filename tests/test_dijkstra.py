from planner.grid import Grid
from planner.dijkstra import dijkstra


def test_finds_shortest_path():
    grid, s, g = Grid.from_ascii([
        "S..#",
        ".#.#",
        ".#..",
        "...G",
    ])
    result = dijkstra(grid, s, g)
    assert result.found
    assert result.cost == 6
    assert result.path[0] == s and result.path[-1] == g


def test_no_path():
    grid, s, g = Grid.from_ascii(["S#G"])
    result = dijkstra(grid, s, g)
    assert not result.found
    assert result.cost is None