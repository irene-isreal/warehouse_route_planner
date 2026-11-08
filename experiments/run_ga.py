import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from planner.genetic import genetic_algorithm
from planner.maps import warehouse_layout
from planner.tsp import cost_matrix, held_karp, nearest_neighbor, pick_points

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
GRID = warehouse_layout(30, 30)
VARIANTS = (("GA (NN-seeded)", True), ("GA (random init)", False))


def convergence(k=16, instance_seed=0, ga_seeds=5):
    D = cost_matrix(GRID, pick_points(GRID, k, instance_seed))
    opt = held_karp(D).cost
    nn = nearest_neighbor(D).cost
    curves = {name: np.array([genetic_algorithm(D, seed=s, seed_with_nn=flag)[1]
                              for s in range(ga_seeds)])
              for name, flag in VARIANTS}
    return opt, nn, curves


def quality_vs_optimal(ks=(8, 12, 16), instances=6, ga_seeds=3):
    rows = []
    for k in ks:
        for inst in range(instances):
            D = cost_matrix(GRID, pick_points(GRID, k, inst))
            hk = held_karp(D)
            nn = nearest_neighbor(D)
            rows.append({"k": k, "instance": inst, "method": "Held-Karp (exact)",
                         "cost": hk.cost, "opt": hk.cost, "time_ms": hk.time_ms})
            rows.append({"k": k, "instance": inst, "method": "Nearest neighbor",
                         "cost": nn.cost, "opt": hk.cost, "time_ms": nn.time_ms})
            for name, flag in VARIANTS:
                for s in range(ga_seeds):
                    res, _ = genetic_algorithm(D, seed=s, seed_with_nn=flag)
                    rows.append({"k": k, "instance": inst, "method": name,
                                 "cost": res.cost, "opt": hk.cost, "time_ms": res.time_ms})
    q = pd.DataFrame(rows)
    q["gap_pct"] = (q["cost"] - q["opt"]) / q["opt"] * 100
    q.to_csv(f"{RESULTS}/ga_quality.csv", index=False)
    return q


def large_instances(ks=(20, 30, 40), instances=4, ga_seeds=2):
    rows = []
    for k in ks:
        for inst in range(instances):
            D = cost_matrix(GRID, pick_points(GRID, k, 100 + inst))
            nn = nearest_neighbor(D).cost
            for name, flag in VARIANTS:
                for s in range(ga_seeds):
                    res, _ = genetic_algorithm(D, seed=s, seed_with_nn=flag)
                    rows.append({"k": k, "instance": inst, "method": name,
                                 "cost": res.cost, "nn_cost": nn, "time_ms": res.time_ms})
    big = pd.DataFrame(rows)
    big["improvement_pct"] = (big["nn_cost"] - big["cost"]) / big["nn_cost"] * 100
    big.to_csv(f"{RESULTS}/ga_large.csv", index=False)
    return big


def plot(opt, nn, curves, q, big):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for name, arr in curves.items():
        gens = np.arange(arr.shape[1])
        ax.plot(gens, np.median(arr, axis=0), label=name)
        ax.fill_between(gens, arr.min(axis=0), arr.max(axis=0), alpha=0.2)
    ax.axhline(opt, color="green", ls="--", label="optimal (Held-Karp)")
    ax.axhline(nn, color="red", ls=":", label="nearest neighbor")
    ax.set_xlabel("generation")
    ax.set_ylabel("best tour cost")
    ax.set_title("Convergence (k=16, 5 GA seeds, band = min/max)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    gap = q.groupby(["k", "method"])["gap_pct"].mean().unstack()
    gap[["Nearest neighbor", "GA (random init)", "GA (NN-seeded)"]].plot.bar(ax=ax, rot=0)
    ax.set_ylabel("mean excess over optimal (%)")
    ax.set_xlabel("number of pick locations k")
    ax.set_title("Solution quality vs the exact optimum")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    t_q = q.groupby(["k", "method"])["time_ms"].median().unstack()
    t_big = big.groupby(["k", "method"])["time_ms"].median().unstack()
    ax.plot(t_q.index, t_q["Held-Karp (exact)"], marker="o", label="Held-Karp (exact)")
    ga_t = pd.concat([t_q["GA (NN-seeded)"], t_big["GA (NN-seeded)"]])
    ax.plot(ga_t.index, ga_t.values, marker="o", label="GA (NN-seeded)")
    ax.set_yscale("log")
    ax.set_xlabel("number of pick locations k")
    ax.set_ylabel("median time (ms, log scale)")
    ax.set_title("Runtime")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    imp = big.groupby(["k", "method"])["improvement_pct"].mean().unstack()
    imp.plot.bar(ax=ax, rot=0)
    ax.set_ylabel("mean improvement over nearest neighbor (%)")
    ax.set_xlabel("number of pick locations k")
    ax.set_title("Large instances (no exact optimum available)")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/ga.png", dpi=150)
    plt.close()
    return gap, imp


if __name__ == "__main__":
    print("Running convergence experiment...")
    opt, nn, curves = convergence()
    print("Running quality-vs-optimal experiment (slowest part)...")
    q = quality_vs_optimal()
    print("Running large-instance experiment...")
    big = large_instances()
    gap, imp = plot(opt, nn, curves, q, big)

    print("\nMean excess over optimal (%):")
    print(gap.round(2).to_string())
    hit = q.groupby(["k", "method"])["gap_pct"].apply(lambda s: (s < 1e-9).mean() * 100).unstack()
    print("\nPercent of runs that found the exact optimum:")
    print(hit.round(1).to_string())
    print("\nMean improvement over nearest neighbor on large instances (%):")
    print(imp.round(2).to_string())
    print("\nMedian GA runtime on large instances (ms):")
    print(big.groupby(["k", "method"])["time_ms"].median().unstack().round(1).to_string())
    print("\nSaved results/ga.png, results/ga_quality.csv, results/ga_large.csv")