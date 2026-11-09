import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

from planner.fuzzy import fuzzy_controller
from planner.navigation import random_world, simulate

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
START, GOAL = (1.5, 1.5), (18.5, 18.5)


def find_failures(sensors, n_obstacles, want=4, max_seed=200):
    found = []
    for seed in range(max_seed):
        world = random_world(seed, n_obstacles, start=START, goal=GOAL)
        log = []
        ep = simulate(world, fuzzy_controller, START, GOAL, sensors=sensors, log=log)
        if ep.outcome == "collision":
            found.append((seed, world, ep, log))
            if len(found) == want:
                break
    return found


if __name__ == "__main__":
    sensors = sys.argv[1] if len(sys.argv) > 1 else "7-ray"
    n_obstacles = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    cases = find_failures(sensors, n_obstacles)

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, (seed, world, ep, log) in zip(axes.ravel(), cases):
        for ox, oy, r in world.obstacles:
            ax.add_patch(Circle((ox, oy), r, color="gray", alpha=0.6))
        xs, ys = zip(*ep.trajectory)
        ax.plot(xs, ys, lw=1.5, label="trajectory")
        ax.plot(xs[-15:], ys[-15:], lw=3, color="orange", label="last 15 steps")
        ax.scatter(xs[-1], ys[-1], marker="x", color="red", s=100, zorder=5, label="end")
        ax.scatter(*START, c="blue", s=60, zorder=5)
        ax.scatter(*GOAL, c="green", s=60, zorder=5)
        ax.set_xlim(0, world.size)
        ax.set_ylim(0, world.size)
        ax.set_aspect("equal")
        ax.set_title(f"seed {seed}: collision at step {ep.steps}")
        ax.legend(fontsize=7, loc="lower right")

        print(f"\n=== seed {seed}, {n_obstacles} obstacles, {sensors}: last 8 steps before collision ===")
        print("(x, y = position when the decision was made; v, w = commands after limits)")
        print(pd.DataFrame(log).tail(8).round(2).to_string(index=False))

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/fuzzy_failures.png", dpi=150)
    plt.close()
    print(f"\nSaved {RESULTS}/fuzzy_failures.png")
