import random
from dataclasses import dataclass, field

from .astar import astar
from .grid import Grid


@dataclass
class Mission:
    success: bool
    steps: int
    replans: int
    nodes: int          # total cells expanded by A* over the whole mission, including the first plan
    trace: list
    frames: list = field(default_factory=list)


def place_hidden(base_grid, start, goal, k, seed):
    """k secretly blocked cells, chosen within one cell of the initial shortest path."""
    rng = random.Random(seed)
    initial = astar(base_grid, start, goal)
    near = set()
    for r, c in initial.path:
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            cell = (r + dr, c + dc)
            if base_grid.is_free(cell) and cell not in (start, goal):
                near.add(cell)
    candidates = sorted(near)
    return set(rng.sample(candidates, min(k, len(candidates))))


def run_mission(base_grid, hidden, start, goal, sense_radius=4, record=False):
    """Follow an A* path, discovering hidden blocks within `sense_radius` (Manhattan distance)
    and replanning with A* whenever a newly discovered block lies on the remaining path."""
    known = Grid(base_grid.rows, base_grid.cols, base_grid.blocked)
    plan = astar(known, start, goal)
    nodes = len(plan.explored)
    replans = 0
    trace, frames = [start], []
    if not plan.found:
        return Mission(False, 0, 0, nodes, trace, frames)

    path, i, pos = plan.path, 0, start
    for _ in range(10 * base_grid.rows * base_grid.cols):
        if pos == goal:
            if record:
                frames.append((pos, frozenset(known.blocked - base_grid.blocked), (), replans))
            return Mission(True, len(trace) - 1, replans, nodes, trace, frames)

        newly = {c for c in hidden
                 if abs(c[0] - pos[0]) + abs(c[1] - pos[1]) <= sense_radius} - known.blocked
        if newly:
            known.blocked |= newly
            if any(c in newly for c in path[i:]):
                plan = astar(known, pos, goal)
                nodes += len(plan.explored)
                replans += 1
                if not plan.found:
                    return Mission(False, len(trace) - 1, replans, nodes, trace, frames)
                path, i = plan.path, 0

        if record:
            frames.append((pos, frozenset(known.blocked - base_grid.blocked),
                           tuple(path[i:]), replans))
        i += 1
        pos = path[i]
        trace.append(pos)

    return Mission(False, len(trace) - 1, replans, nodes, trace, frames)
