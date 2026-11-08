import os

import matplotlib.pyplot as plt
import pandas as pd

from planner.maps import warehouse_layout
from planner.tsp import (branch_and_bound, brute_force, cost_matrix, held_karp,
                         nearest_neighbor, pick_points)

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

ALGOS = {
    "Nearest neighbor": nearest_neighbor,
    "Held-Karp (DP)": held_karp,
    "Branch and bound": branch_and_bound,
    "Brute force": brute_force,
}


def run(ks=(4, 6, 8, 10, 12), seeds=10):
    grid = warehouse_layout(30, 30)
    rows = []
    for k in ks:
        for seed in range(seeds):
            D = cost_matrix(grid, pick_points(grid, k, seed))
            exact_costs = set()
            for name, algo in ALGOS.items():
                if name == "Brute force" and k > 8:
                    continue
                res = algo(D)
                rows.append({"k": k, "seed": seed, "algorithm": name,
                             "cost": res.cost, "time_ms": res.time_ms, "nodes": res.nodes})
                if name != "Nearest neighbor":
                    exact_costs.add(res.cost)
            assert len(exact_costs) == 1, f"exact methods disagree for k={k}, seed={seed}"
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/tsp.csv", index=False)
    return df


def plot(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    times = df.groupby(["k", "algorithm"])["time_ms"].median().reset_index()
    for name, g in times.groupby("algorithm"):
        axes[0].plot(g["k"], g["time_ms"], marker="o", label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of pick locations k")
    axes[0].set_ylabel("median time (ms, log scale)")
    axes[0].set_title("Runtime")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    exact = df[df["algorithm"].isin(["Held-Karp (DP)", "Branch and bound"])]
    nodes = exact.groupby(["k", "algorithm"])["nodes"].median().reset_index()
    for name, g in nodes.groupby("algorithm"):
        axes[1].plot(g["k"], g["nodes"], marker="o", label=name)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("number of pick locations k")
    axes[1].set_ylabel("median states / nodes expanded (log scale)")
    axes[1].set_title("Search effort")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    opt = df[df["algorithm"] == "Held-Karp (DP)"].set_index(["k", "seed"])["cost"]
    nn = df[df["algorithm"] == "Nearest neighbor"].set_index(["k", "seed"])["cost"]
    gap = ((nn - opt) / opt * 100).reset_index(name="gap_pct")
    summary = gap.groupby("k")["gap_pct"].agg(["mean", "max"])
    axes[2].plot(summary.index, summary["mean"], marker="o", label="mean")
    axes[2].plot(summary.index, summary["max"], marker="o", label="worst case")
    axes[2].set_xlabel("number of pick locations k")
    axes[2].set_ylabel("nearest neighbor excess over optimal (%)")
    axes[2].set_title("Quality of the greedy heuristic")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/tsp.png", dpi=150)
    plt.close()
    return times, nodes, summary


if __name__ == "__main__":
    df = run()
    times, nodes, summary = plot(df)
    print("Median time (ms):")
    print(times.pivot(index="k", columns="algorithm", values="time_ms").round(3).to_string())
    print("\nMedian states/nodes (Held-Karp states vs B&B nodes expanded):")
    print(nodes.pivot(index="k", columns="algorithm", values="nodes").to_string())
    print("\nNearest neighbor excess over optimal (%):")
    print(summary.round(2).to_string())
    print("\nAll exact methods agreed on every instance. Saved results/tsp.csv and results/tsp.png")