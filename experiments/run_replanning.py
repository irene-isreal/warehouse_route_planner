import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from planner.astar import astar
from planner.grid import Grid
from planner.maps import warehouse_layout
from planner.replan import place_hidden, run_mission

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
BASE = warehouse_layout(30, 30)
START, GOAL = (1, 1), (28, 28)


def trial(k, seed, radius):
    """One mission. Returns None if the goal is unreachable even with full information."""
    hidden = place_hidden(BASE, START, GOAL, k, seed)
    optimal = astar(Grid(BASE.rows, BASE.cols, BASE.blocked | hidden), START, GOAL)
    if not optimal.found:
        return None
    initial = astar(BASE, START, GOAL)
    no_replan_ok = not any(c in hidden for c in initial.path)
    m = run_mission(BASE, hidden, START, GOAL, radius)
    return {"k": k, "seed": seed, "radius": radius, "replan_success": m.success,
            "no_replan_success": no_replan_ok, "steps": m.steps, "replans": m.replans,
            "nodes": m.nodes, "optimal": optimal.cost,
            "stretch": m.steps / optimal.cost if m.success else float("nan")}


def sweep(values, make_args, seeds):
    rows, skipped = [], 0
    for v in values:
        for seed in range(seeds):
            r = trial(*make_args(v, seed))
            if r is None:
                skipped += 1
            else:
                rows.append(r)
    return pd.DataFrame(rows), skipped


def summarize(df, key):
    g = df.groupby(key)
    return pd.DataFrame({
        "missions": g.size(),
        "no_replan_success_%": g["no_replan_success"].mean() * 100,
        "replan_success_%": g["replan_success"].mean() * 100,
        "mean_replans": g["replans"].mean(),
        "mean_astar_nodes": g["nodes"].mean(),
        "mean_stretch": df[df["replan_success"]].groupby(key)["stretch"].mean(),
    }).round(2)


def plot(by_k, by_r):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(by_k.index, by_k["no_replan_success_%"], marker="o", label="never replan")
    axes[0, 0].plot(by_k.index, by_k["replan_success_%"], marker="o", label="replan on discovery")
    axes[0, 0].set_xlabel("hidden blocked cells near the route")
    axes[0, 0].set_ylabel("mission success (%)")
    axes[0, 0].set_title("Does replanning help?")
    axes[0, 0].legend()

    axes[0, 1].plot(by_k.index, by_k["mean_stretch"], marker="o", color="tab:red")
    axes[0, 1].set_xlabel("hidden blocked cells near the route")
    axes[0, 1].set_ylabel("steps / shortest path on the known map")
    axes[0, 1].set_title("Path stretch (successful missions)")

    axes[1, 0].plot(by_k.index, by_k["mean_astar_nodes"], marker="o", color="tab:green")
    axes[1, 0].set_xlabel("hidden blocked cells near the route")
    axes[1, 0].set_ylabel("A* cells expanded per mission")
    axes[1, 0].set_title("Total planning effort")

    axes[1, 1].plot(by_r.index, by_r["mean_stretch"], marker="o", color="tab:purple")
    axes[1, 1].set_xlabel("sensing radius (cells), 20 hidden blocks")
    axes[1, 1].set_ylabel("steps / shortest path on the known map")
    axes[1, 1].set_title("Effect of how far the robot can sense")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/replanning.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    df_k, skipped_k = sweep((0, 5, 10, 20, 40), lambda k, s: (k, s, 4), seeds)
    df_r, skipped_r = sweep((1, 2, 4, 8), lambda r, s: (20, s, r), seeds)
    df_k.to_csv(f"{RESULTS}/replanning_by_blocks.csv", index=False)
    df_r.to_csv(f"{RESULTS}/replanning_by_radius.csv", index=False)

    by_k, by_r = summarize(df_k, "k"), summarize(df_r, "radius")
    plot(by_k, by_r)
    print(f"Vary the number of hidden blocks (sensing radius 4, {seeds} seeds each):")
    print(by_k.to_string())
    print(f"  (skipped {skipped_k} instances where the goal was unreachable even with full information)")
    print(f"\nVary the sensing radius (20 hidden blocks, {seeds} seeds each):")
    print(by_r.to_string())
    print(f"  (skipped {skipped_r} unreachable instances)")
    print("\nSaved results/replanning.png and two CSVs")
