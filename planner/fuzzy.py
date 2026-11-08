import math

import numpy as np

MAX_RANGE = 3.0   # sensor range (m)
MAX_TURN = 2.0    # rad/s, positive = counter-clockwise (left)
MAX_SPEED = 1.0   # m/s


def trap(x, a, b, c, d):
    """Trapezoidal membership: 0 outside (a, d), rises on [a, b], 1 on [b, c], falls on [c, d].
    a == b or c == d gives a shoulder; b == c gives a triangle."""
    if x < a or x > d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    return (d - x) / (d - c)


# ---- Input fuzzy sets (a, b, c, d) ----
DIST = {                       # front clearance (m)
    "near": (0.0, 0.0, 0.6, 1.2),
    "medium": (0.6, 1.3, 1.3, 2.0),
    "far": (1.5, 2.5, 3.0, 3.0),
}
BALANCE = {                    # left distance minus right distance (m)
    "right_clearer": (-3.0, -3.0, -1.0, 0.0),
    "balanced": (-1.0, 0.0, 0.0, 1.0),
    "left_clearer": (0.0, 1.0, 3.0, 3.0),
}
BEARING = {                    # angle to the goal relative to heading (rad), positive = goal on the left
    "goal_right": (-math.pi, -math.pi, -0.5, 0.0),
    "ahead": (-0.5, 0.0, 0.0, 0.5),
    "goal_left": (0.0, 0.5, math.pi, math.pi),
}

# ---- Output fuzzy sets ----
TURN = {
    "hard_right": (-2.0, -2.0, -1.5, -0.8),
    "right": (-1.5, -0.8, -0.8, 0.0),
    "straight": (-0.5, 0.0, 0.0, 0.5),
    "left": (0.0, 0.8, 0.8, 1.5),
    "hard_left": (0.8, 1.5, 2.0, 2.0),
}
SPEED = {
    "slow": (0.0, 0.0, 0.2, 0.5),
    "medium": (0.3, 0.55, 0.55, 0.8),
    "fast": (0.6, 0.9, 1.0, 1.0),
}

W_UNIVERSE = np.linspace(-MAX_TURN, MAX_TURN, 81)
V_UNIVERSE = np.linspace(0.0, MAX_SPEED, 41)


def _sample(universe, sets):
    return {name: np.array([trap(u, *p) for u in universe]) for name, p in sets.items()}


TURN_MF = _sample(W_UNIVERSE, TURN)
SPEED_MF = _sample(V_UNIVERSE, SPEED)

# Each rule: ({input variable: set name, ...}, (output variable, set name))
RULES = [
    # speed depends on how much room is ahead
    ({"front": "near"}, ("speed", "slow")),
    ({"front": "medium"}, ("speed", "medium")),
    ({"front": "far"}, ("speed", "fast")),
    # obstacle close ahead: turn toward the clearer side
    ({"front": "near", "balance": "left_clearer"}, ("turn", "hard_left")),
    ({"front": "near", "balance": "right_clearer"}, ("turn", "hard_right")),
    ({"front": "near", "balance": "balanced", "bearing": "goal_right"}, ("turn", "hard_right")),
    ({"front": "near", "balance": "balanced", "bearing": "ahead"}, ("turn", "hard_left")),
    ({"front": "near", "balance": "balanced", "bearing": "goal_left"}, ("turn", "hard_left")),
    # obstacle at medium range: gentler turns
    ({"front": "medium", "balance": "left_clearer"}, ("turn", "left")),
    ({"front": "medium", "balance": "right_clearer"}, ("turn", "right")),
    ({"front": "medium", "balance": "balanced", "bearing": "goal_left"}, ("turn", "left")),
    ({"front": "medium", "balance": "balanced", "bearing": "goal_right"}, ("turn", "right")),
    ({"front": "medium", "balance": "balanced", "bearing": "ahead"}, ("turn", "straight")),
    # clear ahead: head for the goal, but drift away from close side obstacles
    ({"front": "far", "bearing": "goal_left"}, ("turn", "left")),
    ({"front": "far", "bearing": "goal_right"}, ("turn", "right")),
    ({"front": "far", "bearing": "ahead"}, ("turn", "straight")),
    ({"front": "far", "balance": "left_clearer"}, ("turn", "left")),
    ({"front": "far", "balance": "right_clearer"}, ("turn", "right")),
]


def _centroid(universe, mu, default):
    total = mu.sum()
    if total < 1e-9:
        return default
    return float((universe * mu).sum() / total)


def fuzzy_controller(left, front, right, bearing):
    """Mamdani fuzzy controller. Returns (speed, turn_rate)."""
    inputs = {
        "front": (DIST, min(max(front, 0.0), MAX_RANGE)),
        "balance": (BALANCE, min(max(left - right, -MAX_RANGE), MAX_RANGE)),
        "bearing": (BEARING, min(max(bearing, -math.pi), math.pi)),
    }
    mu = {var: {name: trap(x, *params) for name, params in sets.items()}
          for var, (sets, x) in inputs.items()}

    agg = {"turn": np.zeros_like(W_UNIVERSE), "speed": np.zeros_like(V_UNIVERSE)}
    mfs = {"turn": TURN_MF, "speed": SPEED_MF}
    for antecedent, (out_var, out_set) in RULES:
        strength = min(mu[var][name] for var, name in antecedent.items())
        if strength > 0:
            agg[out_var] = np.maximum(agg[out_var], np.minimum(strength, mfs[out_var][out_set]))

    turn = _centroid(W_UNIVERSE, agg["turn"], 0.0)
    speed = _centroid(V_UNIVERSE, agg["speed"], 0.3)
    return speed, turn


def reactive_controller(left, front, right, bearing):
    """Baseline: hard thresholds, no fuzzy logic. Returns (speed, turn_rate)."""
    if front < 0.9:
        return 0.2, (1.8 if left > right else -1.8)
    if left < 0.6:
        return 0.5, -1.0
    if right < 0.6:
        return 0.5, 1.0
    return 1.0, max(-MAX_TURN, min(MAX_TURN, 2.0 * bearing))