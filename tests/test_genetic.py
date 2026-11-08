import random

from planner.genetic import genetic_algorithm
from planner.tsp import brute_force, nearest_neighbor, tour_cost


def random_matrix(n, seed):
    rng = random.Random(seed)
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            D[i][j] = D[j][i] = rng.randint(1, 30)
    return D


def test_ga_returns_valid_tour():
    D = random_matrix(10, 1)
    res, _ = genetic_algorithm(D, generations=50, seed=1)
    assert res.tour[0] == 0 and res.tour[-1] == 0
    assert sorted(res.tour[:-1]) == list(range(10))
    assert res.cost == tour_cost(D, res.tour)


def test_ga_best_cost_never_increases():
    D = random_matrix(12, 2)
    _, history = genetic_algorithm(D, generations=60, seed=3, seed_with_nn=False)
    assert all(b <= a for a, b in zip(history, history[1:]))


def test_seeded_ga_never_worse_than_nearest_neighbor():
    for seed in range(5):
        D = random_matrix(14, seed)
        res, _ = genetic_algorithm(D, generations=40, seed=seed)
        assert res.cost <= nearest_neighbor(D).cost


def test_ga_is_deterministic_for_a_given_seed():
    D = random_matrix(12, 5)
    a, _ = genetic_algorithm(D, generations=40, seed=7)
    b, _ = genetic_algorithm(D, generations=40, seed=7)
    assert a.tour == b.tour


def test_ga_is_near_optimal_on_small_instances():
    for seed in range(5):
        D = random_matrix(8, seed)
        opt = brute_force(D).cost
        res, _ = genetic_algorithm(D, seed=seed)
        assert res.cost <= 1.10 * opt