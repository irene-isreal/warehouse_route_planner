import heapq
import time

from .result import SearchResult, reconstruct_path


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal, heuristic=manhattan):
    t0 = time.perf_counter()
    g = {start: 0}
    parent = {}
    # Heap entries: (f, -g, cell). On equal f, the larger g (deeper cell) is expanded first.
    heap = [(heuristic(start, goal), 0, start)]
    done = set()
    explored = []

    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in done:
            continue
        done.add(cur)
        explored.append(cur)

        if cur == goal:
            ms = (time.perf_counter() - t0) * 1000
            return SearchResult(reconstruct_path(parent, start, goal), explored, ms)

        for nxt in grid.neighbors(cur):
            ng = g[cur] + 1
            if ng < g.get(nxt, float("inf")):
                g[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(heap, (ng + heuristic(nxt, goal), -ng, nxt))

    return SearchResult([], explored, (time.perf_counter() - t0) * 1000)