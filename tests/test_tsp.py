import random

from planner.maps import warehouse_layout
from planner.tsp import (branch_and_bound, brute_force, cost_matrix, held_karp,
                         nearest_neighbor, pick_points, tour_cost)


def random_matrix(n, seed):
    rng = random.Random(seed)
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            D[i][j] = D[j][i] = rng.randint(1, 30)
    return D


def check_valid(D, res):
    n = len(D)
    assert res.tour[0] == 0 and res.tour[-1] == 0
    assert sorted(res.tour[:-1]) == list(range(n))
    assert res.cost == tour_cost(D, res.tour)


def test_exact_methods_match_brute_force_on_random_matrices():
    for seed in range(40):
        n = random.Random(seed).randint(2, 8)
        D = random_matrix(n, seed)
        opt = brute_force(D).cost
        hk, bb = held_karp(D), branch_and_bound(D)
        check_valid(D, hk)
        check_valid(D, bb)
        assert hk.cost == opt
        assert bb.cost == opt
        assert nearest_neighbor(D).cost >= opt


def test_tsp_on_warehouse_grid():
    grid = warehouse_layout(30, 30)
    for seed in range(5):
        D = cost_matrix(grid, pick_points(grid, 7, seed))
        opt = brute_force(D).cost
        assert held_karp(D).cost == opt
        assert branch_and_bound(D).cost == opt