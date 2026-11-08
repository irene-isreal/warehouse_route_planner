import heapq
import time

from .result import SearchResult, reconstruct_path


def dijkstra(grid, start, goal):
    t0 = time.perf_counter()
    dist = {start: 0}
    parent = {}
    heap = [(0, start)]
    done = set()
    explored = []

    while heap:
        d, cur = heapq.heappop(heap)
        if cur in done:
            continue
        done.add(cur)
        explored.append(cur)

        if cur == goal:
            ms = (time.perf_counter() - t0) * 1000
            return SearchResult(reconstruct_path(parent, start, goal), explored, ms)

        for nxt in grid.neighbors(cur):
            nd = d + 1
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                parent[nxt] = cur
                heapq.heappush(heap, (nd, nxt))

    return SearchResult([], explored, (time.perf_counter() - t0) * 1000)