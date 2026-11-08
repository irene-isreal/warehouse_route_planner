from planner.fuzzy import fuzzy_controller, trap
from planner.navigation import World, ray_distance, simulate
from planner.fuzzy import reactive_controller


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
    assert abs(ray_distance(0.0, 0.0, 0.0, obstacles) - 4.0) < 1e-9
    assert ray_distance(0.0, 0.0, 1.5707963, obstacles) == 3.0   # misses: capped at max range
    assert ray_distance(0.0, 0.0, 0.0, obstacles, max_range=3.0) == 3.0  # hit is beyond range


def test_both_controllers_reach_goal_in_empty_world():
    world = World(20.0, [])
    for ctrl in (fuzzy_controller, reactive_controller):
        ep = simulate(world, ctrl, (1.0, 1.0), (10.0, 10.0))
        assert ep.outcome == "success"