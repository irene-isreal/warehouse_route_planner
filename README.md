# Warehouse Route Planner

Algorithm study for autonomous warehouse robots: route planning, multi-stop optimisation,
reactive obstacle avoidance and replanning, with experiments that compare each method on
runtime, memory, solution quality and robustness.

Everything runs in simulation (pure Python, NumPy, Matplotlib). Every claim below is backed
by a script in `experiments/` and a CSV or figure in `results/`.

## What is implemented

| Component | Methods | Where |
|---|---|---|
| Shortest paths on a grid | Dijkstra, A* (Manhattan heuristic) | `planner/dijkstra.py`, `planner/astar.py` |
| All-pairs shortest paths | Floyd-Warshall (NumPy), with path reconstruction | `planner/floyd_warshall.py` |
| Multi-stop routing (TSP) | Held-Karp DP, branch and bound, nearest neighbour, brute force | `planner/tsp.py` |
| Heuristic TSP | Genetic algorithm (order crossover, tournament selection, elitism) | `planner/genetic.py` |
| Local obstacle avoidance | Mamdani fuzzy controller (3 variants) vs a threshold-based reactive baseline | `planner/fuzzy.py`, `planner/navigation.py` |
| Replanning | A* replanning when hidden obstacles are discovered | `planner/replan.py` |

## Key findings

- **A\* vs Dijkstra.** On aisle-style warehouse maps up to 160 x 160, A* expanded about as many
  cells as the path length (315 cells vs 17,484 for Dijkstra at 160 x 160), ran about 40x faster
  and used about 17x less peak memory. On a "detour trap" (a wall with one gap) the advantage
  shrinks to about 2x because the heuristic points through the wall.
- **Floyd-Warshall.** Measured log-log slope of runtime vs number of free cells was 3.16 (theory: 3).
  Precomputing all pairs over the whole grid only pays off after roughly 100 (84 cells) to
  20,000 (676 cells) point-to-point queries, so a robot visiting a few shelves is better off
  running A* between the points it needs.
- **TSP, exact methods.** Held-Karp and branch and bound matched brute force on every test
  instance. Branch and bound expanded about 24x fewer nodes than Held-Karp visited states
  (k = 8 to 12), but was slower in wall-clock time in this Python implementation
  (166 ms vs 36 ms at k = 12), because each node does a full matrix reduction.
- **TSP, genetic algorithm.** Found the exact optimum in 100%, 89% and 56% of runs at
  k = 8, 12, 16 pick locations (mean excess 0.00%, 0.19%, 2.16%). At k = 20 to 40, where no exact
  answer is available, it was about 12% shorter than nearest neighbour when seeded with the
  nearest-neighbour tour and 7 to 10% shorter with random initialisation.
- **Fuzzy vs reactive obstacle avoidance.** See `results/final_fuzzy.png` and
  `results/heldout.csv`. The simple threshold controller was more reliable than the fuzzy
  controller in every condition tested. The improved fuzzy controller kept a larger safety
  clearance on worlds both solved, at the cost of longer paths and rougher steering. The
  study also shows how much the sensor layout mattered: moving from 3 to 7 range sensors
  improved both controllers.
- **Replanning.** With hidden blockages near the route, replanning from the current position
  completed every mission where the goal was reachable (guaranteed by the setup, see
  limitations). Path stretch over the optimum on the fully known map was 1.10 with
  20 hidden blocked cells and fell from 1.23 to 1.05 as the sensing radius grew from 1 to 8 cells.

## Reproducing the results

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python -m pytest -q          # unit and property tests
    python run_all.py            # regenerates all tables and figures (several minutes)
    python -m experiments.heldout_check   # held-out evaluation of the fuzzy controllers

Each experiment can also be run on its own, for example `python -m experiments.run_tsp`.
Random seeds are fixed, so the outputs are reproducible; timings vary with the machine.

## Repository layout

    planner/       algorithms and simulation code
    experiments/   one script per experiment (they write to results/)
    tests/         correctness tests, including cross-checks between algorithms
    results/       CSV files and figures produced by the experiments
    run_all.py     runs the tests and all experiments in order

## Limitations

- Everything is simulated: a kinematic point robot in a 2D world, noise-free range sensors,
  no dynamics, no real hardware.
- The fuzzy rules and the reactive baseline's thresholds were hand-designed. The fuzzy
  controller was revised after inspecting failures on development worlds, so a held-out set of
  worlds (seeds 1000 to 1199) is reported separately.
- In the replanning experiment the hidden blockages are placed next to the initial route and
  the robot senses through shelves. Success is guaranteed whenever the goal is reachable
  because the planner starts from an optimistic map, so the informative results are the costs.
- The global planner and the local fuzzy controller are evaluated separately, not integrated.
- Timings come from pure Python on one machine, and are useful for scaling trends, not for
  absolute speed claims.

## Possible extensions

Integrate the A* route with the local controller, add dynamic (moving) obstacles, replace the
fuzzy rule base with a learned policy, and port the planners to a compiled language.

See `docs/REPORT.md` for the full write-up.
