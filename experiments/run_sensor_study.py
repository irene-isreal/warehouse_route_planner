import os

import matplotlib.pyplot as plt
import pandas as pd

from planner.fuzzy import fuzzy_controller, reactive_controller
from planner.navigation import SENSOR_SUITES, random_world, simulate

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
START, GOAL = (1.5, 1.5), (18.5, 18.5)
CONTROLLERS = {"Fuzzy": fuzzy_controller, "Reactive (thresholds)": reactive_controller}


def run(densities=(6, 12, 18), worlds=50):
    rows = []
    for suite in SENSOR_SUITES:
        for n in densities:
            for seed in range(worlds):
                world = random_world(seed, n, start=START, goal=GOAL)
                for name, ctrl in CONTROLLERS.items():
                    ep = simulate(world, ctrl, START, GOAL, sensors=suite)
                    rows.append({"sensors": suite, "obstacles": n, "seed": seed,
                                 "controller": name, "outcome": ep.outcome,
                                 "steps": ep.steps, "path_length": ep.path_length,
                                 "min_clearance": ep.min_clearance,
                                 "smoothness": ep.smoothness})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/sensor_study.csv", index=False)
    return df


def outcome_rates(df):
    rates = pd.crosstab([df["sensors"], df["obstacles"], df["controller"]],
                        df["outcome"], normalize="index") * 100
    return rates.reindex(columns=["success", "collision", "timeout"], fill_value=0.0)


def plot(rates):
    fig, axes = plt.subplots(1, len(SENSOR_SUITES), figsize=(6 * len(SENSOR_SUITES), 4.5),
                             sharey=True)
    for ax, suite in zip(axes, SENSOR_SUITES):
        rates.loc[suite]["success"].unstack("controller").plot.bar(ax=ax, rot=0)
        ax.set_title(f"Sensor suite: {suite}")
        ax.set_xlabel("number of obstacles")
        ax.set_ylabel("success rate (%)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/sensor_study.png", dpi=150)
    plt.close()


def paired(df):
    ok = df[df["outcome"] == "success"]
    wide = ok.pivot_table(index=["sensors", "obstacles", "seed"], columns="controller",
                          values=["min_clearance", "smoothness", "path_length"]).dropna()
    print("\nWorlds where both controllers succeeded:")
    print(wide.groupby(level=["sensors", "obstacles"]).size().to_string())
    print("\nPaired means on those worlds:")
    print(wide.groupby(level=["sensors", "obstacles"]).mean().round(3).to_string())


if __name__ == "__main__":
    df = run()
    rates = outcome_rates(df)
    plot(rates)
    print("Outcome rates (%):")
    print(rates.round(1).to_string())
    paired(df)
    print("\nSaved results/sensor_study.csv and results/sensor_study.png")
