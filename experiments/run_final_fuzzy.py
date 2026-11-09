import math
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from planner.fuzzy import fuzzy_controller, fuzzy_dominant, reactive_controller
from planner.navigation import random_world, simulate

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
START, GOAL = (1.5, 1.5), (18.5, 18.5)
SENSORS = "7-ray"
DENSITIES = (6, 12, 18)
CONTROLLERS = {
    "Fuzzy v1": fuzzy_controller,
    "Fuzzy v2a (final)": fuzzy_dominant,
    "Reactive": reactive_controller,
}


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def run(worlds):
    rows = []
    for n in DENSITIES:
        for seed in range(worlds):
            world = random_world(seed, n, start=START, goal=GOAL)
            for name, ctrl in CONTROLLERS.items():
                ep = simulate(world, ctrl, START, GOAL, sensors=SENSORS, walls=True)
                x, y = ep.trajectory[-1]
                left_world = not (0 <= x <= world.size and 0 <= y <= world.size)
                if ep.outcome == "success":
                    failure = "none"
                elif ep.outcome == "timeout":
                    failure = "timeout"
                else:
                    failure = "left world" if left_world else "obstacle collision"
                rows.append({"obstacles": n, "seed": seed, "controller": name,
                             "outcome": ep.outcome, "failure": failure, "steps": ep.steps,
                             "path_length": ep.path_length, "min_clearance": ep.min_clearance,
                             "smoothness": ep.smoothness})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/final_fuzzy.csv", index=False)
    return df


def plot(df):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.25
    for i, name in enumerate(CONTROLLERS):
        ys, lo, hi = [], [], []
        for n in DENSITIES:
            sub = df[(df["obstacles"] == n) & (df["controller"] == name)]
            k, m = int((sub["outcome"] == "success").sum()), len(sub)
            l, h = wilson(k, m)
            ys.append(100 * k / m)
            lo.append(100 * k / m - 100 * l)
            hi.append(100 * h - 100 * k / m)
        xs = [j + (i - 1) * width for j in range(len(DENSITIES))]
        ax.bar(xs, ys, width, yerr=[lo, hi], capsize=3, label=name)
    ax.set_xticks(range(len(DENSITIES)))
    ax.set_xticklabels(DENSITIES)
    ax.set_xlabel("number of obstacles in a 20 x 20 m world")
    ax.set_ylabel("success rate (%), bars = 95% CI")
    ax.set_ylim(0, 105)
    ax.set_title(f"Final comparison ({SENSORS} sensors, wall sensing on)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/final_fuzzy.png", dpi=150)
    plt.close()


def report(df, worlds):
    print(f"Success rate % [95% Wilson interval], {worlds} worlds per density:")
    table = {}
    for name in CONTROLLERS:
        col = []
        for n in DENSITIES:
            sub = df[(df["obstacles"] == n) & (df["controller"] == name)]
            k, m = int((sub["outcome"] == "success").sum()), len(sub)
            l, h = wilson(k, m)
            col.append(f"{100 * k / m:5.1f} [{100 * l:5.1f}, {100 * h:5.1f}]")
        table[name] = col
    print(pd.DataFrame(table, index=DENSITIES).rename_axis("obstacles").to_string())

    print("\nHow runs ended (counts):")
    print(pd.crosstab([df["obstacles"], df["controller"]], df["failure"]).to_string())

    ok = df[df["outcome"] == "success"]
    wide = ok.pivot_table(index=["obstacles", "seed"], columns="controller",
                          values=["min_clearance", "smoothness", "path_length", "steps"]).dropna()
    print("\nWorlds where ALL controllers succeeded:")
    print(wide.groupby(level="obstacles").size().to_string())
    print("\nMeans on those worlds (steps x 0.1 s = mission time):")
    print(wide.groupby(level="obstacles").mean().round(3).T.to_string())


if __name__ == "__main__":
    worlds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(f"Running {worlds} worlds x {len(DENSITIES)} densities x {len(CONTROLLERS)} controllers...")
    df = run(worlds)
    plot(df)
    report(df, worlds)
    print("\nSaved results/final_fuzzy.csv and results/final_fuzzy.png")
