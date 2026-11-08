import matplotlib.pyplot as plt
import numpy as np


def show(grid, start, goal, result, title=""):
    img = np.zeros((grid.rows, grid.cols))
    for r, c in grid.blocked:
        img[r, c] = 1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, cmap="Greys")

    if result.explored:
        ys, xs = zip(*result.explored)
        ax.scatter(xs, ys, s=18, c="lightskyblue", label="explored")
    if result.path:
        ys, xs = zip(*result.path)
        ax.plot(xs, ys, c="orange", lw=2.5, label="path")

    ax.scatter(start[1], start[0], c="blue", s=90, label="start")
    ax.scatter(goal[1], goal[0], c="green", s=90, label="goal")
    ax.set_title(f"{title} | explored={len(result.explored)} "
                 f"cost={result.cost} time={result.time_ms:.3f} ms")
    ax.legend(loc="upper right", fontsize=8)
    plt.show()