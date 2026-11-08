import math

import pandas as pd

from planner.fuzzy import fuzzy_controller, reactive_controller
from planner.navigation import DT, random_world, simulate

START, GOAL = (1.5, 1.5), (18.5, 18.5)
CONTROLLERS = {"Fuzzy": fuzzy_controller, "Reactive (thresholds)": reactive_controller}


def classify(world, ep):
    """Describe how a collision episode ended, using the last two trajectory points."""
    (x0, y0), (x1, y1) = ep.trajectory[-2], ep.trajectory[-1]
    if not (0 <= x1 <= world.size and 0 <= y1 <= world.size):
        return "left the world", float("nan")
    heading = math.atan2(y1 - y0, x1 - x0)
    speed = math.hypot(x1 - x0, y1 - y0) / DT
    ox, oy, r = min(world.obstacles, key=lambda o: math.hypot(x1 - o[0], y1 - o[1]) - o[2])
    rel = math.atan2(oy - y1, ox - x1) - heading
    rel = abs(math.degrees(math.atan2(math.sin(rel), math.cos(rel))))
    if rel <= 30:
        return "head-on (obstacle within 30 deg of heading)", speed
    if rel <= 100:
        return "side clip (30-100 deg off heading)", speed
    return "rear contact (over 100 deg off heading)", speed


def diagnose(densities=(6, 12, 18), worlds=50):
    rows = []
    for n in densities:
        for seed in range(worlds):
            world = random_world(seed, n, start=START, goal=GOAL)
            for name, ctrl in CONTROLLERS.items():
                ep = simulate(world, ctrl, START, GOAL)
                if ep.outcome == "collision":
                    kind, speed = classify(world, ep)
                    rows.append({"obstacles": n, "controller": name, "kind": kind,
                                 "impact_speed": speed, "steps": ep.steps})
    return pd.DataFrame(rows)


def paired_comparison():
    """Compare clearance and smoothness only on worlds where BOTH controllers succeeded."""
    df = pd.read_csv("results/fuzzy.csv")
    ok = df[df["outcome"] == "success"]
    wide = ok.pivot_table(index=["obstacles", "seed"], columns="controller",
                          values=["min_clearance", "smoothness", "path_length"]).dropna()
    print("\nWorlds where both controllers succeeded:")
    print(wide.groupby(level="obstacles").size().to_string())
    print("\nPaired means (same worlds for both controllers):")
    print(wide.groupby(level="obstacles").mean().round(3).T.to_string())


if __name__ == "__main__":
    collisions = diagnose()
    print("Collision types (counts):")
    print(pd.crosstab([collisions["obstacles"], collisions["controller"]],
                      collisions["kind"]).to_string())
    print("\nMean impact speed (m/s) and mean steps before collision:")
    print(collisions.groupby(["obstacles", "controller"])[["impact_speed", "steps"]]
          .mean().round(2).to_string())
    paired_comparison()