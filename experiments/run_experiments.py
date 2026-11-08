import os

import matplotlib.pyplot as plt
import pandas as pd

from planner.benchmark import ALGORITHMS, measure
from planner.maps import random_obstacles, warehouse_layout, wall_with_gap

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


def exp_scaling():
    rows = []
    for n in (20, 40, 80, 120, 160):
        grid = warehouse_layout(n, n)
        start, goal = (1, 1), (n - 2, n - 2)
        for name, algo in ALGORITHMS.items():
            m = measure(algo, grid, start, goal)
            rows.append({"size": n, "cells": n * n, "algorithm": name, **m})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/scaling.csv", index=False)
    return df


def exp_density(n=60, seeds=30):
    rows = []
    start, goal = (0, 0), (n - 1, n - 1)
    for density in (0.0, 0.1, 0.2, 0.3, 0.35):
        for seed in range(seeds):
            grid = random_obstacles(n, n, density, seed, keep_free=(start, goal))
            for name, algo in ALGORITHMS.items():
                m = measure(algo, grid, start, goal, repeats=5)
                rows.append({"density": density, "seed": seed,
                             "algorithm": name, **m})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/density.csv", index=False)
    return df


def exp_detour():
    rows = []
    for n in (20, 40, 80, 160):
        grid = wall_with_gap(n, n)
        start, goal = (n - 1, 0), (n - 1, n - 1)
        for name, algo in ALGORITHMS.items():
            m = measure(algo, grid, start, goal)
            rows.append({"size": n, "algorithm": name, **m})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/detour.csv", index=False)
    return df


def plot_scaling(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in df.groupby("algorithm"):
        axes[0].plot(g["cells"], g["explored"], marker="o", label=name)
        axes[1].plot(g["cells"], g["time_ms"], marker="o", label=name)
    axes[0].set_ylabel("cells explored")
    axes[1].set_ylabel("median time (ms)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("grid cells (log scale)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Warehouse layouts: cost of search vs map size")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/scaling.png", dpi=150)
    plt.close()


def plot_density(df):
    solved = df[df["found"]]
    summary = solved.groupby(["density", "algorithm"])[["explored", "time_ms"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in summary.groupby("algorithm"):
        axes[0].plot(g["density"], g["explored"], marker="o", label=name)
        axes[1].plot(g["density"], g["time_ms"], marker="o", label=name)
    axes[0].set_ylabel("mean cells explored (solved maps)")
    axes[1].set_ylabel("mean time (ms, solved maps)")
    for ax in axes:
        ax.set_xlabel("obstacle density")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Random obstacles (60x60): effect of density")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/density.png", dpi=150)
    plt.close()

    success = df[df["algorithm"] == "A*"].groupby("density")["found"].mean()
    print("\nFraction of random maps that were solvable:")
    print(success.to_string())


def plot_detour(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in df.groupby("algorithm"):
        ax.plot(g["size"] ** 2, g["explored"], marker="o", label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grid cells (log scale)")
    ax.set_ylabel("cells explored")
    ax.set_title("Detour trap: wall with a gap at the top")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/detour.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    scaling = exp_scaling()
    print("\n== Scaling ==")
    print(scaling[["size", "algorithm", "cost", "explored", "time_ms", "peak_kb"]].round(3).to_string(index=False))
    plot_scaling(scaling)

    density = exp_density()
    plot_density(density)

    detour = exp_detour()
    print("\n== Detour trap ==")
    print(detour[["size", "algorithm", "cost", "explored", "time_ms", "peak_kb"]].round(3).to_string(index=False))
    plot_detour(detour)

    print(f"\nSaved CSVs and plots in ./{RESULTS}/")