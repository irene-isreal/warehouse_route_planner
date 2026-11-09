import math

from planner.fuzzy import fuzzy_controller, reactive_controller
from planner.navigation import World, ray_distance, read_sensors, simulate


def test_ray_hits_walls():
    assert abs(ray_distance(1.0, 5.0, 0.0, [], max_range=30.0, world_size=20.0) - 19.0) < 1e-9
    assert abs(ray_distance(1.0, 5.0, math.pi, [], max_range=30.0, world_size=20.0) - 1.0) < 1e-9
    assert abs(ray_distance(1.0, 5.0, math.pi / 2, [], max_range=30.0, world_size=20.0) - 15.0) < 1e-6


def test_walls_ignored_unless_world_size_given():
    assert ray_distance(1.0, 5.0, math.pi, [], max_range=30.0) == 30.0


def test_front_sensor_sees_wall_ahead():
    _, front, _ = read_sensors([], 18.5, 10.0, 0.0, "3-ray", world_size=20.0)
    assert abs(front - 1.5) < 1e-9
    _, front_blind, _ = read_sensors([], 18.5, 10.0, 0.0, "3-ray")
    assert front_blind == 3.0


def test_both_controllers_reach_goal_with_wall_sensing():
    world = World(20.0, [])
    for ctrl in (fuzzy_controller, reactive_controller):
        ep = simulate(world, ctrl, (5.0, 5.0), (15.0, 15.0), sensors="7-ray", walls=True)
        assert ep.outcome == "success"
