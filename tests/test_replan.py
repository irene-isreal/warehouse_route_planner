from planner.astar import astar
from planner.grid import Grid
from planner.maps import warehouse_layout
from planner.replan import place_hidden, run_mission


def test_no_hidden_blocks_follows_optimal_path():
    base = warehouse_layout(20, 20)
    m = run_mission(base, set(), (1, 1), (18, 18))
    opt = astar(base, (1, 1), (18, 18))
    assert m.success and m.replans == 0
    assert m.steps == opt.cost
    assert m.nodes == len(opt.explored)


def test_detours_around_a_blocked_cell():
    base = Grid(5, 5)
    m = run_mission(base, {(0, 2)}, (0, 0), (0, 4))
    assert m.success
    assert m.replans == 1
    assert (0, 2) not in m.trace
    assert m.steps == 6


def test_reports_failure_when_goal_is_cut_off():
    base = Grid(1, 5)
    m = run_mission(base, {(0, 2)}, (0, 0), (0, 4))
    assert not m.success
    assert m.replans == 1


def test_never_enters_hidden_cells_and_never_beats_optimal():
    base = warehouse_layout(20, 20)
    for seed in range(20):
        hidden = place_hidden(base, (1, 1), (18, 18), 15, seed)
        m = run_mission(base, hidden, (1, 1), (18, 18))
        assert not (set(m.trace) & hidden)
        if m.success:
            true_grid = Grid(base.rows, base.cols, base.blocked | hidden)
            opt = astar(true_grid, (1, 1), (18, 18))
            assert opt.found and m.steps >= opt.cost
