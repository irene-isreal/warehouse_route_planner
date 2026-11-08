from planner.astar import astar
from planner.maps import warehouse_layout, wall_with_gap


def test_warehouse_layout_is_solvable():
    for n in (20, 40, 80):
        grid = warehouse_layout(n, n)
        assert astar(grid, (1, 1), (n - 2, n - 2)).found


def test_wall_with_gap_forces_detour():
    n = 20
    grid = wall_with_gap(n, n)
    result = astar(grid, (n - 1, 0), (n - 1, n - 1))
    assert result.found
    assert result.cost > n - 1  # a straight line would cost n - 1