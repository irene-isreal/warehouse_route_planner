from planner.fuzzy import fuzzy_controller, reactive_controller, trap
from planner.navigation import World, ray_distance, read_sensors, simulate


def test_trapezoid_membership_values():
    assert trap(0.5, 0, 1, 1, 2) == 0.5
    assert trap(1.0, 0, 1, 1, 2) == 1.0
    assert trap(2.0, 0, 1, 1, 2) == 0.0
    assert trap(-1.0, 0, 1, 1, 2) == 0.0
    assert trap(0.0, 0, 0, 1, 2) == 1.0   # left shoulder


def test_turns_toward_the_clearer_side():
    _, w_left = fuzzy_controller(left=3.0, front=0.4, right=1.0, bearing=0.0)
    _, w_right = fuzzy_controller(left=1.0, front=0.4, right=3.0, bearing=0.0)
    assert w_left > 0.5
    assert w_right < -0.5


def test_slows_down_near_obstacles():
    v_free, _ = fuzzy_controller(3.0, 3.0, 3.0, 0.0)
    v_near, _ = fuzzy_controller(3.0, 0.4, 3.0, 0.0)
    assert v_free > 0.75
    assert v_near < 0.35


def test_steers_toward_goal_in_free_space():
    _, w_ahead = fuzzy_controller(3.0, 3.0, 3.0, 0.0)
    _, w_goal_left = fuzzy_controller(3.0, 3.0, 3.0, 1.0)
    _, w_goal_right = fuzzy_controller(3.0, 3.0, 3.0, -1.0)
    assert abs(w_ahead) < 0.05
    assert w_goal_left > 0.3
    assert w_goal_right < -0.3


def test_ray_distance():
    obstacles = [(5.0, 0.0, 1.0)]
    # obstacle surface is 4 m away; use a longer range than the default 3 m sensor
    assert abs(ray_distance(0.0, 0.0, 0.0, obstacles, max_range=10.0) - 4.0) < 1e-9
    assert ray_distance(0.0, 0.0, 0.0, obstacles, max_range=3.0) == 3.0     # hit is beyond range
    assert ray_distance(0.0, 0.0, 1.5707963, obstacles, max_range=10.0) == 10.0  # ray misses


def test_seven_ray_suite_sees_obstacle_beside_robot():
    obstacles = [(0.0, 1.0, 0.4)]   # directly to the left of a robot at the origin heading +x
    left3, _, _ = read_sensors(obstacles, 0.0, 0.0, 0.0, "3-ray")
    left7, _, _ = read_sensors(obstacles, 0.0, 0.0, 0.0, "7-ray")
    assert left3 == 3.0                 # the 45-degree ray misses it
    assert abs(left7 - 0.6) < 1e-9      # the 90-degree ray hits its surface at 0.6 m


def test_seven_ray_suite_sees_slightly_off_axis_obstacle_ahead():
    obstacles = [(2.0, 0.5, 0.3)]       # ahead and 0.5 m to the left of the centreline
    _, front3, _ = read_sensors(obstacles, 0.0, 0.0, 0.0, "3-ray")
    _, front7, _ = read_sensors(obstacles, 0.0, 0.0, 0.0, "7-ray")
    assert front3 == 3.0                # the centre ray passes beside it
    assert front7 < 2.0                 # the +20 degree ray hits it


def test_both_controllers_reach_goal_in_empty_world():
    world = World(20.0, [])
    for ctrl in (fuzzy_controller, reactive_controller):
        for suite in ("3-ray", "7-ray"):
            ep = simulate(world, ctrl, (1.0, 1.0), (10.0, 10.0), sensors=suite)
            assert ep.outcome == "success"




def test_logging_does_not_change_the_episode():
    world = World(20.0, [(5.0, 5.0, 1.0)])
    plain = simulate(world, fuzzy_controller, (1.0, 1.0), (10.0, 10.0), sensors="7-ray")
    log = []
    logged = simulate(world, fuzzy_controller, (1.0, 1.0), (10.0, 10.0), sensors="7-ray", log=log)
    assert plain.outcome == logged.outcome
    assert plain.steps == logged.steps
    assert plain.trajectory == logged.trajectory
    assert len(log) == logged.steps


import random

from planner.fuzzy import fuzzy_dominant, fuzzy_dominant_side, make_fuzzy


def test_make_fuzzy_with_switches_off_reproduces_the_original_controller():
    rng = random.Random(0)
    plain = make_fuzzy(False, False)
    for _ in range(200):
        args = (rng.uniform(0, 3), rng.uniform(0, 3), rng.uniform(0, 3), rng.uniform(-3.14, 3.14))
        a, b = fuzzy_controller(*args), plain(*args)
        assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9


def test_original_cancels_conflicting_turns_and_dominant_variant_commits():
    # Goal is on the left, but an obstacle is close on the left (0.45 m).
    _, w_original = fuzzy_controller(left=0.45, front=3.0, right=3.0, bearing=0.69)
    _, w_dominant = fuzzy_dominant(left=0.45, front=3.0, right=3.0, bearing=0.69)
    assert abs(w_original) < 1e-6
    assert w_dominant < -0.5


def test_side_aware_speed_slows_down_beside_a_close_obstacle():
    v_original, _ = fuzzy_controller(left=0.45, front=3.0, right=3.0, bearing=0.0)
    v_side, _ = fuzzy_dominant_side(left=0.45, front=3.0, right=3.0, bearing=0.0)
    assert v_original > 0.75
    assert v_side < 0.35


def test_variants_reach_goal_in_empty_world():
    world = World(20.0, [])
    for ctrl in (fuzzy_dominant, fuzzy_dominant_side):
        ep = simulate(world, ctrl, (1.0, 1.0), (10.0, 10.0), sensors="7-ray")
        assert ep.outcome == "success"
