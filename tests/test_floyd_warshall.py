import random

from planner.dijkstra import dijkstra
from planner.floyd_warshall import floyd_warshall
from planner.maps import random_obstacles, warehouse_layout


def _check(grid, seed):
    ap = floyd_warshall(grid)
    rng = random.Random(seed)
    for _ in range(60):
        a, b = rng.choice(ap.cells), rng.choice(ap.cells)
        res = dijkstra(grid, a, b)
        assert ap.distance(a, b) == res.cost
        path = ap.path(a, b)
        if res.found:
            assert path[0] == a and path[-1] == b
            assert len(path) - 1 == res.cost
            for u, v in zip(path, path[1:]):
                assert abs(u[0] - v[0]) + abs(u[1] - v[1]) == 1
        else:
            assert path == []


def test_matches_dijkstra_on_warehouse_layout():
    _check(warehouse_layout(20, 20), seed=1)


def test_matches_dijkstra_with_unreachable_cells():
    # density 0.35 leaves many cells unreachable from each other
    _check(random_obstacles(20, 20, 0.35, seed=3), seed=2)