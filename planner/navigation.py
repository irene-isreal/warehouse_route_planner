import math
import random
from dataclasses import dataclass

from .fuzzy import MAX_RANGE, MAX_TURN

ROBOT_RADIUS = 0.3
GOAL_TOLERANCE = 0.4
DT = 0.1

# Ray angles (radians relative to heading, positive = left) that feed each of the three
# readings the controllers receive. Each reading is the minimum over its rays.
SENSOR_SUITES = {
    "3-ray": {
        "left": (math.radians(45),),
        "front": (0.0,),
        "right": (-math.radians(45),),
    },
    "7-ray": {
        "left": (math.radians(45), math.radians(90)),
        "front": (-math.radians(20), 0.0, math.radians(20)),
        "right": (-math.radians(45), -math.radians(90)),
    },
}


@dataclass
class World:
    size: float
    obstacles: list  # (x, y, radius)


@dataclass
class Episode:
    outcome: str          # "success", "collision" (includes leaving the world) or "timeout"
    steps: int
    path_length: float
    min_clearance: float  # smallest gap between robot edge and any obstacle (m)
    smoothness: float     # mean |change in turn rate| per step (lower = smoother)
    trajectory: list


def random_world(seed, n_obstacles, size=20.0, start=(1.5, 1.5), goal=(18.5, 18.5)):
    """Random circular obstacles, kept away from start/goal and at least 1 m apart."""
    rng = random.Random(seed)
    obstacles = []
    for _ in range(2000):
        if len(obstacles) == n_obstacles:
            break
        x, y, r = rng.uniform(1, size - 1), rng.uniform(1, size - 1), rng.uniform(0.5, 1.2)
        if math.hypot(x - start[0], y - start[1]) < r + 1.5:
            continue
        if math.hypot(x - goal[0], y - goal[1]) < r + 1.5:
            continue
        if any(math.hypot(x - ox, y - oy) < r + orad + 1.0 for ox, oy, orad in obstacles):
            continue
        obstacles.append((x, y, r))
    return World(size, obstacles)


def ray_distance(x, y, angle, obstacles, max_range=MAX_RANGE, world_size=None):
    """Distance along a ray from (x, y) to the nearest circular obstacle, capped at max_range.
    If world_size is given, the four walls of the square arena [0, world_size]^2 also block the ray."""
    dx, dy = math.cos(angle), math.sin(angle)
    best = max_range
    for ox, oy, r in obstacles:
        fx, fy = x - ox, y - oy
        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - r * r
        disc = b * b - c
        if disc < 0:
            continue
        t = -b - math.sqrt(disc)
        if 0 <= t < best:
            best = t
    if world_size is not None:
        for d, p in ((dx, x), (dy, y)):
            if abs(d) > 1e-12:
                t = (world_size - p) / d if d > 0 else (0.0 - p) / d
                if 0 <= t < best:
                    best = t
    return best


def read_sensors(obstacles, x, y, theta, suite="3-ray", world_size=None):
    """Return (left, front, right) readings for the chosen sensor suite."""
    angles = SENSOR_SUITES[suite]
    return tuple(
        min(ray_distance(x, y, theta + a, obstacles, world_size=world_size) for a in angles[key])
        for key in ("left", "front", "right")
    )


def simulate(world, controller, start, goal, max_steps=600, sensors="3-ray", log=None, walls=False):
    """Run one episode. If `log` is a list, one dict per step is appended to it.
    If walls is True, the arena walls are visible to the range sensors."""
    x, y = start
    theta = math.atan2(goal[1] - y, goal[0] - x)
    path_length, min_clear = 0.0, float("inf")
    prev_w, dw_sum = 0.0, 0.0
    trajectory = [(x, y)]
    outcome, step = "timeout", 0
    wall_size = world.size if walls else None

    for step in range(1, max_steps + 1):
        left, front, right = read_sensors(world.obstacles, x, y, theta, sensors, wall_size)

        bearing = math.atan2(goal[1] - y, goal[0] - x) - theta
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))  # wrap to [-pi, pi]

        v, w = controller(left, front, right, bearing)
        dist_goal = math.hypot(goal[0] - x, goal[1] - y)
        v = min(v, max(0.15, 0.8 * dist_goal))   # slow down near the goal (same rule for all controllers)
        w = max(-MAX_TURN, min(MAX_TURN, w))

        if log is not None:
            log.append({"step": step, "x": x, "y": y, "left": left, "front": front,
                        "right": right, "bearing": bearing, "v": v, "w": w})

        theta += w * DT
        x += v * math.cos(theta) * DT
        y += v * math.sin(theta) * DT
        path_length += v * DT
        dw_sum += abs(w - prev_w)
        prev_w = w
        trajectory.append((x, y))

        clearance = min((math.hypot(x - ox, y - oy) - r for ox, oy, r in world.obstacles),
                        default=float("inf")) - ROBOT_RADIUS
        min_clear = min(min_clear, clearance)

        if clearance <= 0 or not (0 <= x <= world.size and 0 <= y <= world.size):
            outcome = "collision"
            break
        if math.hypot(goal[0] - x, goal[1] - y) < GOAL_TOLERANCE:
            outcome = "success"
            break

    return Episode(outcome, step, path_length, min_clear, dw_sum / max(step, 1), trajectory)
 