import time

import numpy as np


class AllPairs:
    """Result of Floyd-Warshall: distance matrix plus next-hop matrix for paths."""

    def __init__(self, cells, index, dist, nxt, time_ms):
        self.cells = cells      # list of free cells; position in list = matrix index
        self.index = index      # cell -> matrix index
        self.dist = dist
        self.nxt = nxt
        self.time_ms = time_ms

    def distance(self, a, b):
        d = self.dist[self.index[a], self.index[b]]
        return None if np.isinf(d) else int(d)

    def path(self, a, b):
        i, j = self.index[a], self.index[b]
        if self.nxt[i, j] == -1:
            return []
        out = [a]
        while i != j:
            i = int(self.nxt[i, j])
            out.append(self.cells[i])
        return out


def floyd_warshall(grid):
    t0 = time.perf_counter()

    cells = [(r, c) for r in range(grid.rows) for c in range(grid.cols)
             if grid.is_free((r, c))]
    index = {cell: i for i, cell in enumerate(cells)}
    V = len(cells)

    dist = np.full((V, V), np.inf)
    nxt = np.full((V, V), -1, dtype=np.int32)
    for i, cell in enumerate(cells):
        dist[i, i] = 0
        nxt[i, i] = i
        for nb in grid.neighbors(cell):
            j = index[nb]
            dist[i, j] = 1
            nxt[i, j] = j

    # For each intermediate cell k, try routing every pair (i, j) through k.
    for k in range(V):
        via = dist[:, k, None] + dist[None, k, :]
        better = via < dist
        dist = np.where(better, via, dist)
        nxt = np.where(better, nxt[:, k, None], nxt)

    return AllPairs(cells, index, dist, nxt, (time.perf_counter() - t0) * 1000)