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

# ---------------- Version 2 variants (ablation switches) ----------------

# Speed rules that also look at the nearest side reading ("side" = min(left, right)).
RULES_SPEED_SIDE = [
    ({"front": "near"}, "slow"),
    ({"side": "near"}, "slow"),
    ({"front": "medium", "side": "medium"}, "medium"),
    ({"front": "medium", "side": "far"}, "medium"),
    ({"front": "far", "side": "medium"}, "medium"),
    ({"front": "far", "side": "far"}, "fast"),
]


def make_fuzzy(dominant_turn=False, side_aware_speed=False):
    """Build a fuzzy controller. With both switches off it reproduces fuzzy_controller exactly.

    dominant_turn: if left-turn and right-turn rules are both active, keep only the stronger
        side (ties go to the side with more room) before taking the centroid.
    side_aware_speed: speed also depends on the nearest side reading.
    """
    turn_rules = [(a, s) for a, (v, s) in RULES if v == "turn"]
    if side_aware_speed:
        speed_rules = RULES_SPEED_SIDE
    else:
        speed_rules = [(a, s) for a, (v, s) in RULES if v == "speed"]

    def controller(left, front, right, bearing):
        front_c = min(max(front, 0.0), MAX_RANGE)
        side_c = min(max(min(left, right), 0.0), MAX_RANGE)
        balance_c = min(max(left - right, -MAX_RANGE), MAX_RANGE)
        bearing_c = min(max(bearing, -math.pi), math.pi)
        mu = {
            "front": {n: trap(front_c, *p) for n, p in DIST.items()},
            "side": {n: trap(side_c, *p) for n, p in DIST.items()},
            "balance": {n: trap(balance_c, *p) for n, p in BALANCE.items()},
            "bearing": {n: trap(bearing_c, *p) for n, p in BEARING.items()},
        }

        turn_act = {name: 0.0 for name in TURN}
        for antecedent, out_set in turn_rules:
            s = min(mu[var][name] for var, name in antecedent.items())
            turn_act[out_set] = max(turn_act[out_set], s)

        speed_act = {name: 0.0 for name in SPEED}
        for antecedent, out_set in speed_rules:
            s = min(mu[var][name] for var, name in antecedent.items())
            speed_act[out_set] = max(speed_act[out_set], s)

        if dominant_turn:
            left_s = max(turn_act["left"], turn_act["hard_left"])
            right_s = max(turn_act["right"], turn_act["hard_right"])
            if left_s > 0 and right_s > 0:
                keep_left = left_s > right_s or (left_s == right_s and left >= right)
                for name in (("right", "hard_right") if keep_left else ("left", "hard_left")):
                    turn_act[name] = 0.0

        w_mu = np.zeros_like(W_UNIVERSE)
        for name, s in turn_act.items():
            if s > 0:
                w_mu = np.maximum(w_mu, np.minimum(s, TURN_MF[name]))
        v_mu = np.zeros_like(V_UNIVERSE)
        for name, s in speed_act.items():
            if s > 0:
                v_mu = np.maximum(v_mu, np.minimum(s, SPEED_MF[name]))

        return _centroid(V_UNIVERSE, v_mu, 0.3), _centroid(W_UNIVERSE, w_mu, 0.0)

    return controller


fuzzy_dominant = make_fuzzy(dominant_turn=True, side_aware_speed=False)
fuzzy_dominant_side = make_fuzzy(dominant_turn=True, side_aware_speed=True)
