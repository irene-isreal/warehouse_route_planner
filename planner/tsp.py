import heapq
import itertools
import random
import time
from dataclasses import dataclass

from .astar import astar

INF = float("inf")


@dataclass
class TourResult:
    tour: list        # city indices, starts and ends at 0, e.g. [0, 3, 1, 2, 0]
    cost: float
    time_ms: float
    nodes: int = 0    # B&B nodes expanded, Held-Karp states, or permutations tried


def pick_points(grid, k, seed, depot=(1, 1)):
    """Depot plus k random pick locations: free cells next to a shelf."""
    rng = random.Random(seed)
    candidates = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            cell = (r, c)
            if not grid.is_free(cell) or cell == depot:
                continue
            adjacent = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if any(nb in grid.blocked for nb in adjacent):
                candidates.append(cell)
    return [depot] + rng.sample(candidates, k)


def cost_matrix(grid, points):
    """Walking distance between every pair of points, computed with A*."""
    n = len(points)
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            res = astar(grid, points[i], points[j])
            d = res.cost if res.found else INF
            D[i][j] = D[j][i] = d
    return D


def tour_cost(D, tour):
    return sum(D[a][b] for a, b in zip(tour, tour[1:]))


def _ms(t0):
    return (time.perf_counter() - t0) * 1000


def nearest_neighbor(D):
    t0 = time.perf_counter()
    n = len(D)
    tour, left = [0], set(range(1, n))
    while left:
        nxt = min(left, key=lambda j: D[tour[-1]][j])
        tour.append(nxt)
        left.remove(nxt)
    tour.append(0)
    return TourResult(tour, tour_cost(D, tour), _ms(t0), n)


def brute_force(D):
    t0 = time.perf_counter()
    n = len(D)
    best_tour, best_cost, tried = None, INF, 0
    for perm in itertools.permutations(range(1, n)):
        tour = [0, *perm, 0]
        c = tour_cost(D, tour)
        tried += 1
        if c < best_cost:
            best_tour, best_cost = tour, c
    return TourResult(best_tour, best_cost, _ms(t0), tried)


def held_karp(D):
    t0 = time.perf_counter()
    n = len(D)
    if n == 1:
        return TourResult([0, 0], 0, _ms(t0), 0)

    m = n - 1  # cities 1..n-1 are stored as bits 0..m-1
    dp = [[INF] * m for _ in range(1 << m)]      # dp[mask][j]: start at 0, visit mask, end at j+1
    parent = [[-1] * m for _ in range(1 << m)]
    for j in range(m):
        dp[1 << j][j] = D[0][j + 1]

    states = 0
    for mask in range(1, 1 << m):
        for j in range(m):
            if not (mask >> j) & 1 or dp[mask][j] == INF:
                continue
            states += 1
            for k in range(m):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                c = dp[mask][j] + D[j + 1][k + 1]
                if c < dp[nm][k]:
                    dp[nm][k] = c
                    parent[nm][k] = j

    full = (1 << m) - 1
    best_j = min(range(m), key=lambda j: dp[full][j] + D[j + 1][0])
    best = dp[full][best_j] + D[best_j + 1][0]
    if best == INF:
        return TourResult([], INF, _ms(t0), states)

    order, mask, j = [], full, best_j
    while j != -1:
        order.append(j + 1)
        pj = parent[mask][j]
        mask ^= 1 << j
        j = pj
    tour = [0] + order[::-1] + [0]
    return TourResult(tour, best, _ms(t0), states)


def _reduce(M):
    """Row- and column-reduce M in place; return the total amount subtracted."""
    n = len(M)
    total = 0
    for i in range(n):
        m = min(M[i])
        if m != INF and m > 0:
            total += m
            M[i] = [x - m if x != INF else INF for x in M[i]]
    for j in range(n):
        m = min(M[i][j] for i in range(n))
        if m != INF and m > 0:
            total += m
            for i in range(n):
                if M[i][j] != INF:
                    M[i][j] -= m
    return total


def branch_and_bound(D):
    """Best-first (least-cost) branch and bound with a reduced-cost-matrix lower bound.
    Starts from the nearest-neighbor tour as the initial upper bound."""
    t0 = time.perf_counter()
    n = len(D)

    nn = nearest_neighbor(D)
    best_tour, best_cost = nn.tour, nn.cost

    M = [[INF if i == j else D[i][j] for j in range(n)] for i in range(n)]
    root_bound = _reduce(M)

    counter = itertools.count()
    heap = [(root_bound, next(counter), [0], M)]
    expanded = 0

    while heap:
        bound, _, path, M = heapq.heappop(heap)
        if bound >= best_cost:
            break  # best-first order: every remaining node is at least this bad
        expanded += 1
        i = path[-1]

        for j in range(n):
            if j in path or M[i][j] == INF:
                continue

            if len(path) + 1 == n:  # j is the last city: close the tour and score it exactly
                tour = path + [j, 0]
                c = tour_cost(D, tour)
                if c < best_cost:
                    best_cost, best_tour = c, tour
                continue

            child = [row[:] for row in M]
            for k in range(n):
                child[i][k] = INF  # leave city i only once
                child[k][j] = INF  # enter city j only once
            child[j][0] = INF      # cannot return to the depot before visiting everyone
            child_bound = bound + M[i][j] + _reduce(child)
            if child_bound < best_cost:
                heapq.heappush(heap, (child_bound, next(counter), path + [j], child))

    return TourResult(best_tour, best_cost, _ms(t0), expanded)