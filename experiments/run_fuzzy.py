import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

from planner.fuzzy import fuzzy_controller, reactive_controller
from planner.navigation import random_world, simulate

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
START, GOAL = (1.5, 1.5), (18.5, 18.5)
CONTROLLERS = {"Fuzzy": fuzzy_controller, "Reactive (thresholds)": reactive_controller}


def run(densities=(6, 12, 18), worlds=50):
    rows = []
    for n in densities:
        for seed in range(worlds):
            world = random_world(seed, n, start=START, goal=GOAL)
            for name, ctrl in CONTROLLERS.items():
                ep = simulate(world, ctrl, START, GOAL)
                rows.append({"obstacles": n, "seed": seed, "controller": name,
                             "outcome": ep.outcome, "steps": ep.steps,
                             "path_length": ep.path_length,
                             "min_clearance": ep.min_clearance,
                             "smoothness": ep.smoothness})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/fuzzy.csv", index=False)
    return df


def outcome_rates(df):
    rates = pd.crosstab([df["obstacles"], df["controller"]], df["outcome"], normalize="index") * 100
    return rates.reindex(columns=["success", "collision", "timeout"], fill_value=0.0)


def plot(df, rates):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    rates["success"].unstack("controller").plot.bar(ax=axes[0], rot=0)
    axes[0].set_ylabel("success rate (%)")
    axes[0].set_xlabel("number of obstacles in a 20 x 20 m world")
    axes[0].set_title("Success rate (50 random worlds per density)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.3, axis="y")

    world = random_world(0, 12, start=START, goal=GOAL)
    ax = axes[1]
    for ox, oy, r in world.obstacles:
        ax.add_patch(Circle((ox, oy), r, color="gray", alpha=0.6))
    for name, ctrl in CONTROLLERS.items():
        ep = simulate(world, ctrl, START, GOAL)
        xs, ys = zip(*ep.trajectory)
        ax.plot(xs, ys, label=f"{name}: {ep.outcome}", lw=2)
    ax.scatter(*START, c="blue", s=80, zorder=5)
    ax.scatter(*GOAL, c="green", s=80, zorder=5)
    ax.set_xlim(0, world.size)
    ax.set_ylim(0, world.size)
    ax.set_aspect("equal")
    ax.set_title("Example world (12 obstacles, seed 0)")
    ax.legend(fontsize=8, loc="lower right")

    ok = df[df["outcome"] == "success"]
    smooth = ok.groupby(["obstacles", "controller"])["smoothness"].mean().unstack("controller")
    smooth.plot.bar(ax=axes[2], rot=0)
    axes[2].set_ylabel("mean |change in turn rate| per step (lower = smoother)")
    axes[2].set_xlabel("number of obstacles")
    axes[2].set_title("Steering smoothness (successful runs only)")
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/fuzzy.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    df = run()
    rates = outcome_rates(df)
    plot(df, rates)

    print("Outcome rates (%):")
    print(rates.round(1).to_string())
    ok = df[df["outcome"] == "success"]
    print("\nSuccessful runs only (means):")
    print(ok.groupby(["obstacles", "controller"])[["path_length", "min_clearance", "smoothness"]]
          .mean().round(3).to_string())
    print("\nSaved results/fuzzy.csv and results/fuzzy.png")