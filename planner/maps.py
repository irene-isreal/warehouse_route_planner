import random

from .grid import Grid


def warehouse_layout(rows, cols, shelf_h=2, shelf_w=4, aisle=2):
    """Aisle-style layout: blocks of shelves separated by aisles. Scales with size.
    Start (1, 1) and goal (rows-2, cols-2) are always free."""
    blocked = set()
    r = aisle
    while r + shelf_h <= rows - aisle:
        c = aisle
        while c + shelf_w <= cols - aisle:
            for rr in range(r, r + shelf_h):
                for cc in range(c, c + shelf_w):
                    blocked.add((rr, cc))
            c += shelf_w + aisle
        r += shelf_h + aisle
    return Grid(rows, cols, blocked)


def random_obstacles(rows, cols, density, seed, keep_free=()):
    """Each cell is blocked with probability `density`. Reproducible via `seed`."""
    rng = random.Random(seed)
    blocked = {(r, c) for r in range(rows) for c in range(cols)
               if rng.random() < density}
    for cell in keep_free:
        blocked.discard(cell)
    return Grid(rows, cols, blocked)


def wall_with_gap(rows, cols):
    """Vertical wall down the middle with a gap only at the top two rows.
    Start at bottom-left and goal at bottom-right force a long detour."""
    mid = cols // 2
    blocked = {(r, mid) for r in range(2, rows)}
    return Grid(rows, cols, blocked)