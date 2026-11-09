import math

import pandas as pd

from experiments.run_final_fuzzy import CONTROLLERS, DENSITIES, GOAL, SENSORS, START, wilson
from planner.navigation import random_world, simulate


def mcnemar_exact(b, c):
    """Two-sided exact p-value for discordant pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_table(df, a="Fuzzy v1", b="Fuzzy v2a (final)"):
    wide = df.pivot_table(index=["obstacles", "seed"], columns="controller",
                          values="outcome", aggfunc="first")
    ok = wide == "success"
    rows = []
    for n in DENSITIES:
        sub = ok.loc[n]
        only_a = int((sub[a] & ~sub[b]).sum())
        only_b = int((~sub[a] & sub[b]).sum())
        rows.append({"obstacles": n, "only_v1_succeeded": only_a,
                     "only_v2a_succeeded": only_b,
                     "exact_p": round(mcnemar_exact(only_a, only_b), 4)})
    return pd.DataFrame(rows).set_index("obstacles")


def run_heldout(seeds=range(1000, 1200)):
    rows = []
    for n in DENSITIES:
        for seed in seeds:
            world = random_world(seed, n, start=START, goal=GOAL)
            for name, ctrl in CONTROLLERS.items():
                ep = simulate(world, ctrl, START, GOAL, sensors=SENSORS, walls=True)
                rows.append({"obstacles": n, "seed": seed, "controller": name,
                             "outcome": ep.outcome})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    dev = pd.read_csv("results/final_fuzzy.csv")
    print("Development worlds (seeds 0-199): v1 vs v2a, paired exact test")
    print(paired_table(dev).to_string())

    print("\nRunning held-out worlds (seeds 1000-1199), a few minutes...")
    held = run_heldout()
    held.to_csv("results/heldout.csv", index=False)

    print("\nHeld-out success rate % [95% Wilson interval]:")
    table = {}
    for name in CONTROLLERS:
        col = []
        for n in DENSITIES:
            sub = held[(held["obstacles"] == n) & (held["controller"] == name)]
            k, m = int((sub["outcome"] == "success").sum()), len(sub)
            lo, hi = wilson(k, m)
            col.append(f"{100 * k / m:5.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}]")
        table[name] = col
    print(pd.DataFrame(table, index=DENSITIES).rename_axis("obstacles").to_string())

    print("\nHow held-out runs ended (counts):")
    print(pd.crosstab([held["obstacles"], held["controller"]], held["outcome"]).to_string())

    print("\nHeld-out worlds: v1 vs v2a, paired exact test")
    print(paired_table(held).to_string())
    print("\nSaved results/heldout.csv")
