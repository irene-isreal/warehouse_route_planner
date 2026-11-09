import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from planner.astar import astar
from planner.grid import Grid
from planner.maps import warehouse_layout
from planner.replan import place_hidden, run_mission

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
BASE = warehouse_layout(30, 30)
START, GOAL = (1, 1), (28, 28)
SHELF_IMG = np.zeros((BASE.rows, BASE.cols))
for r, c in BASE.blocked:
    SHELF_IMG[r, c] = 1.0


def scatter_cells(ax, cells, **kw):
    cells = list(cells)
    if cells:
        ax.scatter([c for _, c in cells], [r for r, _ in cells], **kw)


def find_scenario(k=20, radius=4, min_replans=3):
    for seed in range(300):
        hidden = place_hidden(BASE, START, GOAL, k, seed)
        if not astar(Grid(BASE.rows, BASE.cols, BASE.blocked | hidden), START, GOAL).found:
            continue
        m = run_mission(BASE, hidden, START, GOAL, radius, record=True)
        if m.success and m.replans >= min_replans:
            return seed, hidden, m
    raise SystemExit("No suitable scenario found; lower min_replans in find_scenario().")


def render(ax, frame, hidden, step):
    pos, found, path, replans = frame
    ax.clear()
    ax.imshow(SHELF_IMG, cmap="Greys", vmin=0, vmax=1.6)
    scatter_cells(ax, hidden - set(found), marker="s", s=30, facecolors="none",
                  edgecolors="tomato", linewidths=1)
    scatter_cells(ax, found, marker="s", s=45, c="red")
    if path:
        ax.plot([c for _, c in path], [r for r, _ in path], c="orange", lw=2)
    ax.scatter(START[1], START[0], marker="o", s=80, facecolors="none", edgecolors="blue")
    ax.scatter(GOAL[1], GOAL[0], marker="*", s=140, c="green", zorder=4)
    ax.scatter(pos[1], pos[0], c="blue", s=90, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"step {step} | replans: {replans} | discovered blocks: {len(found)}/{len(hidden)}",
                 fontsize=10)


if __name__ == "__main__":
    seed, hidden, m = find_scenario()
    frames = m.frames
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    anim = FuncAnimation(fig, lambda i: render(ax, frames[i], hidden, i),
                         frames=len(frames), interval=120)
    anim.save(f"{RESULTS}/replay.gif", writer=PillowWriter(fps=8), dpi=80)
    render(ax, frames[-1], hidden, len(frames) - 1)
    fig.savefig(f"{RESULTS}/replay_final.png", dpi=150)
    print(f"Scenario seed {seed}: {m.steps} steps, {m.replans} replans, {m.nodes} A* cells expanded.")
    print("Saved results/replay.gif and results/replay_final.png")
