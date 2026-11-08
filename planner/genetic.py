import random
import time

from .tsp import TourResult, brute_force, nearest_neighbor


def _tour_len(D, perm):
    """Length of the tour 0 -> perm -> 0."""
    total = D[0][perm[0]]
    for a, b in zip(perm, perm[1:]):
        total += D[a][b]
    return total + D[perm[-1]][0]


def order_crossover(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    used = set(child[a:b + 1])
    remaining = iter(x for x in p2 if x not in used)
    for i in range(n):
        if child[i] is None:
            child[i] = next(remaining)
    return child


def mutate(perm, rng):
    i, j = sorted(rng.sample(range(len(perm)), 2))
    if rng.random() < 0.5:
        perm[i], perm[j] = perm[j], perm[i]
    else:
        perm[i:j + 1] = perm[i:j + 1][::-1]


def _tournament(pop, fit, size, rng):
    best = min(rng.sample(range(len(pop)), size), key=lambda i: fit[i])
    return pop[best]


def genetic_algorithm(D, pop_size=100, generations=300, mutation_rate=0.2,
                      tournament_size=3, elite=2, seed=0, seed_with_nn=True):
    """Returns (TourResult, history). history[g] is the best cost at the start of generation g,
    with one extra final entry."""
    t0 = time.perf_counter()
    n = len(D)
    if n <= 4:  # too few cities for a GA to make sense
        res = brute_force(D)
        return res, [res.cost]

    rng = random.Random(seed)
    cities = list(range(1, n))

    pop = []
    if seed_with_nn:
        pop.append(nearest_neighbor(D).tour[1:-1])
    while len(pop) < pop_size:
        p = cities[:]
        rng.shuffle(p)
        pop.append(p)

    fit = [_tour_len(D, p) for p in pop]
    history = []
    evaluations = len(pop)

    for _ in range(generations):
        order = sorted(range(pop_size), key=lambda i: fit[i])
        history.append(fit[order[0]])
        new_pop = [pop[i][:] for i in order[:elite]]
        new_fit = [fit[i] for i in order[:elite]]
        while len(new_pop) < pop_size:
            child = order_crossover(_tournament(pop, fit, tournament_size, rng),
                                    _tournament(pop, fit, tournament_size, rng), rng)
            if rng.random() < mutation_rate:
                mutate(child, rng)
            new_pop.append(child)
            new_fit.append(_tour_len(D, child))
            evaluations += 1
        pop, fit = new_pop, new_fit

    best_i = min(range(pop_size), key=lambda i: fit[i])
    history.append(fit[best_i])
    tour = [0] + pop[best_i] + [0]
    ms = (time.perf_counter() - t0) * 1000
    return TourResult(tour, fit[best_i], ms, evaluations), history