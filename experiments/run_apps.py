import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from planner.astar import astar
from planner.floyd_warshall import floyd_warshall
from planner.maps import warehouse_layout

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


def run():
    rng = random.Random(0)
    rows = []
    for n in (10, 15, 20, 25, 30):
        grid = warehouse_layout(n, n)
        ap = floyd_warshall(grid)
        V = len(ap.cells)
        pairs = [(rng.choice(ap.cells), rng.choice(ap.cells)) for _ in range(100)]
        query_ms = float(np.mean([astar(grid, a, b).time_ms for a, b in pairs]))
        rows.append({
            "size": n,
            "V": V,
            "floyd_warshall_ms": ap.time_ms,
            "astar_query_ms": query_ms,
            "breakeven_queries": ap.time_ms / query_ms,
            "matrix_kb": ap.dist.nbytes / 1024,
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/apsp.csv", index=False)
    return df


def plot(df):
    slope = np.polyfit(np.log(df["V"]), np.log(df["floyd_warshall_ms"]), 1)[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(df["V"], df["floyd_warshall_ms"], marker="o", label="Floyd-Warshall (all pairs)")
    axes[0].plot(df["V"], df["astar_query_ms"], marker="o", label="A* (one query)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("free cells V (log scale)")
    axes[0].set_ylabel("time (ms, log scale)")
    axes[0].set_title(f"Measured growth: FW log-log slope = {slope:.2f}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["V"], df["breakeven_queries"], marker="o", color="tab:red")
    axes[1].set_xlabel("free cells V")
    axes[1].set_ylabel("A* queries before FW precompute pays off")
    axes[1].set_title("Break-even number of queries")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/apsp.png", dpi=150)
    plt.close()
    return slope


if __name__ == "__main__":
    df = run()
    print(df.round(3).to_string(index=False))
    slope = plot(df)
    print(f"\nLog-log slope of Floyd-Warshall time vs V: {slope:.2f} (theory: 3)")
    print(f"Saved results/apsp.csv and results/apsp.png")