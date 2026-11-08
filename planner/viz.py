import matplotlib.pyplot as plt
import numpy as np


def draw(ax, grid, start, goal, result, title):
    img = np.zeros((grid.rows, grid.cols))
    for r, c in grid.blocked:
        img[r, c] = 1
    ax.imshow(img, cmap="Greys")
    if result.explored:
        ys, xs = zip(*result.explored)
        ax.scatter(xs, ys, s=14, c="lightskyblue")
    if result.path:
        ys, xs = zip(*result.path)
        ax.plot(xs, ys, c="orange", lw=2.5)
    ax.scatter(start[1], start[0], c="blue", s=80)
    ax.scatter(goal[1], goal[0], c="green", s=80)
    ax.set_title(f"{title}\nexplored={len(result.explored)}  cost={result.cost}  "
                 f"time={result.time_ms:.3f} ms", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def compare(grid, start, goal, results):
    """results: dict of {name: SearchResult}"""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    for ax, (name, res) in zip(axes, results.items()):
        draw(ax, grid, start, goal, res, name)
    plt.tight_layout()
    plt.show()