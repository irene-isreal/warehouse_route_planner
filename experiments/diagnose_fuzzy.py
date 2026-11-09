import math
import sys

import pandas as pd

from planner.fuzzy import fuzzy_controller, reactive_controller
from planner.navigation import DT, random_world, read_sensors, simulate

START, GOAL = (1.5, 1.5), (18.5, 18.5)
CONTROLLERS = {"Fuzzy": fuzzy_controller, "Reactive (thresholds)": reactive_controller}
SEEN = 1.0   # a ray reporting less than this (m) counts as "the controller could see something"


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
        return "head-on", speed
    if rel <= 100:
        return "side clip", speed
    return "rear contact", speed


def pre_impact_reading(world, ep, sensors):
    """Smallest left/front/right reading one step before impact (heading approximated from motion)."""
    if len(ep.trajectory) < 3:
        return float("nan")
    (xa, ya), (xb, yb) = ep.trajectory[-3], ep.trajectory[-2]
    theta = math.atan2(yb - ya, xb - xa)
    return min(read_sensors(world.obstacles, xb, yb, theta, sensors))


def diagnose(sensors, densities=(6, 12, 18), worlds=50):
    coll, tout = [], []
    for n in densities:
        for seed in range(worlds):
            world = random_world(seed, n, start=START, goal=GOAL)
            for name, ctrl in CONTROLLERS.items():
                ep = simulate(world, ctrl, START, GOAL, sensors=sensors)
                if ep.outcome == "collision":
                    kind, speed = classify(world, ep)
                    coll.append({"obstacles": n, "seed": seed, "controller": name,
                                 "kind": kind, "impact_speed": speed, "steps": ep.steps,
                                 "min_reading": pre_impact_reading(world, ep, sensors)})
                elif ep.outcome == "timeout":
                    x, y = ep.trajectory[-1]
                    tout.append({"obstacles": n, "seed": seed, "controller": name,
                                 "dist_to_goal": math.hypot(GOAL[0] - x, GOAL[1] - y)})
    return pd.DataFrame(coll), pd.DataFrame(tout)


if __name__ == "__main__":
    sensors = sys.argv[1] if len(sys.argv) > 1 else "7-ray"
    coll, tout = diagnose(sensors)
    print(f"=== Sensor suite: {sensors} ===")

    if coll.empty:
        print("No collisions.")
    else:
        print("\nCollision types (counts):")
        print(pd.crosstab([coll["obstacles"], coll["controller"]], coll["kind"]).to_string())
        print("\nMean impact speed (m/s), mean steps before collision:")
        print(coll.groupby(["obstacles", "controller"])[["impact_speed", "steps"]].mean().round(2).to_string())
        print(f"\nMean of the smallest sensor reading one step before impact (m), "
              f"and % of collisions where no ray saw anything within {SEEN} m:")
        g = coll.groupby(["obstacles", "controller"])["min_reading"]
        print(pd.DataFrame({"mean_min_reading": g.mean().round(2),
                            "pct_unseen": g.apply(lambda s: (s >= SEEN).mean() * 100).round(1)}).to_string())

    if tout.empty:
        print("\nNo timeouts.")
    else:
        print("\nTimeouts: count and mean final distance to goal (m):")
        print(tout.groupby(["obstacles", "controller"])["dist_to_goal"].agg(["count", "mean"]).round(2).to_string())
