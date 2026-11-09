import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from planner.fuzzy import (fuzzy_controller, fuzzy_dominant, fuzzy_dominant_side,
                           reactive_controller)
from planner.navigation import random_world, simulate

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
START, GOAL = (1.5, 1.5), (18.5, 18.5)
SENSORS = "7-ray"
CONTROLLERS = {
    "Fuzzy v1 (original)": fuzzy_controller,
    "Fuzzy v2a (+dominant turn)": fuzzy_dominant,
    "Fuzzy v2b (+side-aware speed)": fuzzy_dominant_side,
    "Reactive (thresholds)": reactive_controller,
}


def run(densities=(6, 12, 18), worlds=50):
    rows = []
    for n in densities:
        for seed in range(worlds):
            world = random_world(seed, n, start=START, goal=GOAL)
            for name, ctrl in CONTROLLERS.items():
                ep = simulate(world, ctrl, START, GOAL, sensors=SENSORS)
                x, y = ep.trajectory[-1]
                rows.append({"obstacles": n, "seed": seed, "controller": name,
                             "outcome": ep.outcome, "steps": ep.steps,
                             "path_length": ep.path_length,
                             "min_clearance": ep.min_clearance,
                             "smoothness": ep.smoothness,
                             "left_world": not (0 <= x <= world.size and 0 <= y <= world.size)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/fuzzy_variants.csv", index=False)
    return df


def outcome_rates(df):
    rates = pd.crosstab([df["obstacles"], df["controller"]], df["outcome"],
                        normalize="index") * 100
    return rates.reindex(columns=["success", "collision", "timeout"], fill_value=0.0)


def plot(rates):
    table = rates["success"].unstack("controller")[list(CONTROLLERS)]
    ax = table.plot.bar(figsize=(9, 4.5), rot=0)
    ax.set_ylabel("success rate (%)")
    ax.set_xlabel("number of obstacles")
    ax.set_ylim(0, 105)
    ax.set_title(f"Fuzzy variants vs reactive baseline ({SENSORS} sensors)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/fuzzy_variants.png", dpi=150)
    plt.close()


def paired(df):
    ok = df[df["outcome"] == "success"]
    wide = ok.pivot_table(index=["obstacles", "seed"], columns="controller",
                          values=["min_clearance", "smoothness", "path_length", "steps"]).dropna()
    print("\nWorlds where ALL four controllers succeeded:")
    print(wide.groupby(level="obstacles").size().to_string())
    print("\nMeans on those worlds (steps x 0.1 s = mission time):")
    print(wide.groupby(level="obstacles").mean().round(3).T.to_string())


if __name__ == "__main__":
    worlds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    df = run(worlds=worlds)
    rates = outcome_rates(df)
    plot(rates)
    print(f"Outcome rates (%), {worlds} worlds per density:")
    print(rates.round(1).to_string())
    print("\nRuns that ended by leaving the world (no sensor can see the boundary):")
    print(df.groupby(["obstacles", "controller"])["left_world"].sum().unstack("controller").to_string())
    paired(df)
    print("\nSaved results/fuzzy_variants.csv and results/fuzzy_variants.png")
