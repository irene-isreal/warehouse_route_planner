# Route Planning, Multi-Stop Optimisation and Local Navigation for Warehouse Robots: An Experimental Study

## Abstract

This project studies the software stack of an autonomous warehouse robot in simulation: global
route planning (Dijkstra, A*, Floyd-Warshall), multi-stop routing as a travelling-salesman
problem (Held-Karp dynamic programming, branch and bound, a genetic algorithm), local obstacle
avoidance (a Mamdani fuzzy controller against a threshold-based reactive baseline) and
replanning when blockages are discovered. Every component is implemented from scratch and
evaluated with reproducible experiments. A* expanded roughly path-length many cells on
warehouse layouts, against area-many for Dijkstra, but its advantage shrank to about 2x when
the heuristic was misleading. Computing all-pairs shortest paths over a whole grid was
impractical compared with targeted A* queries. A genetic algorithm found the exact TSP optimum
in most small instances and improved on nearest neighbour by about 12% on larger ones when seeded with the nearest-neighbour tour (7 to 10% with a random start). In local
navigation, the simple reactive baseline was more reliable than my fuzzy controller in every
condition tested; the fuzzy controller kept larger safety clearances at the cost of longer
paths and more stalls, and an attempted improvement did not show a statistically established
gain on held-out worlds. Replanning from the current position completed all reachable
missions with modest path stretch.

## 1. Introduction

A warehouse robot has to decide how to travel between points, in which order to visit several
shelves, how to avoid obstacles it senses on the way, and what to do when a planned route turns
out to be blocked. This project implements one method or more for each of these four problems
and asks the same question of each: how well does it perform, and what does it cost in time,
memory or reliability? The emphasis is on measurement and on reporting results that did not
go the way I expected.

## 2. Methods

### 2.1 Environments

- **Grid maps.** 4-connected grids with unit step cost. Three families: an aisle-style
  warehouse layout (2 x 4 shelf blocks separated by 2-cell aisles with a 2-cell border), random
  obstacle maps at a chosen density, and a "detour trap" (a wall across the middle with a gap
  only at the top).
- **Continuous arena.** A 20 x 20 m square arena with circular obstacles (radius 0.5 to 1.2 m,
  at least 1 m apart) and a circular robot of radius 0.3 m moving with limits of 1 m/s and
  2 rad/s, simulated at 0.1 s steps for at most 600 steps. The robot starts at (1.5, 1.5) and
  must reach within 0.4 m of (18.5, 18.5).

### 2.2 Shortest paths

Dijkstra and A* (Manhattan heuristic, ties broken towards deeper nodes) are implemented with a
binary heap and lazy deletion. Floyd-Warshall is implemented with NumPy (one array operation
per intermediate cell) with a next-hop matrix for path reconstruction. Correctness is checked
against each other on hundreds of random maps, including unreachable goals. Search cost is
measured as cells expanded, median wall-clock time over 20 repeats, and peak memory from
`tracemalloc`.

### 2.3 Multi-stop routing

A mission is a depot plus k pick locations (free cells next to a shelf) in a 30 x 30 warehouse
layout; the distance matrix is computed with A*. Solvers: brute force (verification only),
Held-Karp dynamic programming, best-first branch and bound with a reduced-cost-matrix lower
bound and a nearest-neighbour initial upper bound, and nearest neighbour as a greedy baseline.
The genetic algorithm uses permutation encoding, order crossover, swap or segment-reversal
mutation (rate 0.2), tournament selection (size 3), elitism (2), population 100 and 300
generations. It is run both seeded with the nearest-neighbour tour and with a purely random
initial population.

### 2.4 Local navigation

A Mamdani fuzzy controller takes three inputs: front clearance, the left-minus-right clearance
difference, and the bearing to the goal. It outputs a speed and a turn rate, using trapezoidal
membership functions, 18 rules, min for rule strength, max aggregation and centroid
defuzzification. Two variants were derived during the study: v2a keeps only the stronger
turning direction when left- and right-turn rules are both active, and v2b (an ablation) also
makes speed depend on the nearest side reading. The baseline is a threshold controller: slow
down and turn towards the more open side if the front reading is under 0.9 m, steer away from a
side reading under 0.6 m, otherwise drive at full speed towards the goal. Both controllers read
the same three range values (left, front, right), each the minimum over a set of simulated rays
with a 3 m range.

### 2.5 Replanning

On a 30 x 30 warehouse layout the robot follows an A* path while sensing blocked cells within a
Manhattan radius. When a newly discovered blocked cell lies on its remaining path, it replans
with A* from its current position. Hidden blockages are placed within one cell of the initial
shortest path. The baseline never replans.

### 2.6 Evaluation protocol

Random seeds are fixed. Fuzzy controllers were developed on worlds with seeds 0 to 199 and
finally evaluated on 200 unseen worlds (seeds 1000 to 1199) per obstacle density. Success rates
are reported with 95% Wilson intervals, and controller pairs are compared with an exact
McNemar test on paired outcomes. Six such tests were run without multiple-comparison correction.
All experiments are in `experiments/` and regenerate with `python run_all.py` (the held-out
evaluation is `python -m experiments.heldout_check`).

## 3. Results

### 3.1 A* versus Dijkstra

On warehouse layouts of increasing size (start and goal in opposite corners):

| Grid | Dijkstra cells | A* cells | Dijkstra ms | A* ms | Dijkstra peak KB | A* peak KB |
|---|---|---|---|---|---|---|
| 20 x 20 | 300 | 35 | 0.619 | 0.094 | 32.6 | 9.2 |
| 40 x 40 | 1,164 | 75 | 2.530 | 0.219 | 128.1 | 29.6 |
| 80 x 80 | 4,420 | 155 | 10.072 | 0.454 | 587.8 | 62.3 |
| 120 x 120 | 9,988 | 235 | 22.347 | 0.700 | 1,612.6 | 65.5 |
| 160 x 160 | 17,484 | 315 | 41.174 | 1.030 | 2,708.4 | 155.0 |

Dijkstra's work grows with the area of the map, while A* expanded one more cell than the path
length (cost + 1) in every case, because these layouts have an open border route on which the
Manhattan heuristic is almost exact. This is A*'s best case. On the detour trap, where the
straight line to the goal is blocked, A* expanded 209, 819, 3,239 and 12,879 cells for grids of
20, 40, 80 and 160 against 382, 1,562, 6,322 and 25,442 for Dijkstra: an advantage of about 1.8x to 2x instead of 9x to 55x on the open layouts (in cells expanded). On 60 x 60 random-obstacle maps the fraction of solvable
instances fell from 100% at densities 0 and 0.1 to 93% at 0.2, 53% at 0.3 and 20% at 0.35.

### 3.2 All-pairs shortest paths

| Free cells | Floyd-Warshall ms | Mean A* query ms | Queries to break even | Matrix KB |
|---|---|---|---|---|
| 84 | 2.2 | 0.022 | 98 | 55 |
| 177 | 15.5 | 0.034 | 452 | 245 |
| 304 | 77.7 | 0.048 | 1,632 | 722 |
| 505 | 569.1 | 0.065 | 8,823 | 1,992 |
| 676 | 1,501.3 | 0.075 | 20,029 | 3,570 |

The log-log slope of Floyd-Warshall time against the number of cells was 3.16, close to the
theoretical 3. The break-even column is the practical result: precomputing all pairs only pays
off after hundreds to tens of thousands of queries, and memory grows quadratically. A robot that
needs distances between a handful of shelves is better served by A* between those points.

### 3.3 Multi-stop routing (exact methods)

Median over 10 instances per k (brute force only up to k = 8):

| k | Held-Karp ms | Branch and bound ms | Brute force ms | B&B nodes | Held-Karp states |
|---|---|---|---|---|---|
| 4 | 0.026 | 0.117 | 0.022 | 6 | 32 |
| 6 | 0.167 | 0.609 | 0.666 | 19 | 192 |
| 8 | 1.052 | 2.642 | 42.962 | 44.5 | 1,024 |
| 10 | 6.397 | 19.219 | n/a | 212 | 5,120 |
| 12 | 36.345 | 165.660 | n/a | 1,023.5 | 24,576 |

All exact methods agreed with each other on every instance, and with brute force wherever it
was run. Branch and bound expanded about 24 times fewer nodes than Held-Karp visited states
for k from 8 to 12, yet was slower in wall-clock time, because each node copies and reduces an
n x n matrix in Python. This result is specific to my implementation and to the node-versus-state
unit of work; it should not be read as a general statement that dynamic programming beats branch
and bound. Nearest neighbour was fast but its mean excess over the optimum grew from 7.5% (k = 4)
to 19.2% (k = 12), with a worst case of 40%.

### 3.4 Genetic algorithm

Against the exact optimum (6 instances x 3 GA seeds per setting):

| k | Mean excess, NN-seeded GA | Mean excess, random-init GA | Runs reaching the optimum (both) |
|---|---|---|---|
| 8 | 0.00% | 0.00% | 100% |
| 12 | 0.19% | 0.29% | 88.9% |
| 16 | 2.16% | 2.31% | 55.6% |

Where no exact answer is available (k = 20, 30, 40; 4 instances x 2 seeds), the seeded GA was
11.7%, 12.2% and 12.7% shorter than nearest neighbour; the random-initialisation GA was 10.3%,
7.6% and 7.5% shorter. Runtime stayed between 275 and 339 ms because the evaluation budget is
fixed. The GA hyperparameters were not tuned, and I did not test whether the random-start gap
closes with more generations.

### 3.5 Local navigation: fuzzy control against a reactive baseline

The study went through four iterations, each motivated by a diagnosis.

1. **Initial comparison (3 range rays, 50 worlds per density).** The reactive controller
   succeeded in 92%, 74% and 66% of worlds (6, 12, 18 obstacles) and the fuzzy controller in 64%,
   38% and 16%. Almost every collision, for both controllers, was a side clip. The reactive
   controller's mean impact speed was exactly its top speed (1.00 m/s), which is consistent with
   its sensors not seeing the obstacle at all.
2. **Seven rays.** Adding rays at +-20 and +-90 degrees (same interface, both controllers)
   raised reactive success to 100%, 100% and 98% and fuzzy success to 84%, 72% and 50%.
3. **Conflicting rules.** A flight recorder on failing runs showed the fuzzy controller
   commanding a turn rate of about zero while goal-seeking and side-clearance rules asked for
   opposite turns; centroid defuzzification averaged them into "go straight" (evaluating the
   controller on the logged inputs returned 0.87 m/s and a turn rate of about 0). Variant v2a
   fixes this by letting only the stronger turn direction act. Variant v2b (side-aware speed) did
   not improve success on 50 worlds and made missions longer and steering rougher.
4. **Wall sensing.** The simulator initially had invisible arena walls and some failures were
   runs leaving the arena. Letting the range sensors detect the walls (for all controllers)
   removed this failure: no run in the 200-world development evaluation left the arena (the held-out runs were not classified this way).

Final evaluation, development worlds (seeds 0 to 199, 200 per density), success rate with 95%
Wilson interval:

| Obstacles | Fuzzy v1 | Fuzzy v2a | Reactive |
|---|---|---|---|
| 6 | 88.5% [83.3, 92.2] | 92.5% [88.0, 95.4] | 100.0% [98.1, 100.0] |
| 12 | 76.5% [70.2, 81.8] | 85.5% [80.0, 89.7] | 99.5% [97.2, 99.9] |
| 18 | 58.0% [51.1, 64.6] | 70.0% [63.3, 75.9] | 97.5% [94.3, 98.9] |

Held-out worlds (seeds 1000 to 1199, 200 per density):

| Obstacles | Fuzzy v1 | Fuzzy v2a | Reactive |
|---|---|---|---|
| 6 | 90.0% [85.1, 93.4] | 90.0% [85.1, 93.4] | 100.0% [98.1, 100.0] |
| 12 | 77.5% [71.2, 82.7] | 84.0% [78.3, 88.4] | 99.0% [96.4, 99.7] |
| 18 | 65.0% [58.2, 71.3] | 70.0% [63.3, 75.9] | 99.0% [96.4, 99.7] |

How the runs ended (obstacle collisions / timeouts), development and held-out:

| Obstacles | v1 (dev) | v2a (dev) | v1 (held-out) | v2a (held-out) |
|---|---|---|---|---|
| 6 | 23 / 0 | 7 / 8 | 19 / 1 | 8 / 12 |
| 12 | 45 / 2 | 14 / 15 | 41 / 4 | 12 / 20 |
| 18 | 80 / 4 | 34 / 26 | 69 / 1 | 29 / 31 |

The reactive controller had at most 4 collisions and 1 timeout in any condition. The paired
exact test of v1 against v2a gave p = 0.115, 0.011 and 0.005 on the development worlds, but
p = 1.00, 0.060 and 0.28 on the held-out worlds (6, 12, 18 obstacles). Variant v2a therefore cut
obstacle collisions by roughly 58 to 71% in both sets, but mostly by turning collisions into
timeouts (the robot not reaching the goal within 600 steps); the net success gain is not
established on unseen worlds. I did not investigate the cause of the timeouts; a stalled or
oscillating robot in cluttered regions is a plausible explanation that I have not verified.

On development worlds that all three controllers solved (171, 138 and 92 worlds), the fuzzy
variant v2a kept more clearance than the reactive controller (0.78, 0.41 and 0.27 m against
0.45, 0.21 and 0.18 m) but travelled paths that were 5%, 11% and 15% longer, needed 20%, 27%
and 32% more steps, and had rougher steering (mean absolute change in turn rate per step 0.169,
0.289 and 0.376 against 0.106, 0.195 and 0.284). The original fuzzy controller v1 had the
smoothest steering of the three (0.041, 0.074, 0.099) and clearance close to the reactive
controller at 18 obstacles (0.170 against 0.175 m).

### 3.6 Replanning

With 200 missions per setting (sensing radius 4 cells):

| Hidden blocked cells | Success without replanning | Mean replans | Mean A* cells expanded | Path stretch |
|---|---|---|---|---|
| 0 | 100% | 0.00 | 55 | 1.00 |
| 5 | 9.5% | 1.58 | 105 | 1.03 |
| 10 | 1.5% | 2.84 | 141 | 1.05 |
| 20 | 0% | 5.22 | 203 | 1.10 |
| 40 | 0% | 9.25 | 289 | 1.14 |

(At 40 blocks, 13 of 200 instances were skipped because the goal was unreachable even with full
information.) Replanning completed every mission, which is guaranteed by the setup: the
planner starts from an optimistic map, so a path always exists while the goal is reachable. The
no-replanning baseline fails by construction, since blockages are placed next to the route.
The informative quantities are the costs: stretch is the distance travelled divided by the
shortest path on the fully known map, and each replan expanded roughly 25 to 32 cells on average (against 55 for the initial plan; replans start nearer the goal, so this is not a like-for-like comparison). A longer sensing range reduced both stretch and replans: with 20 hidden blocks, stretch fell from 1.23 (radius 1) to 1.17,
1.10 and 1.05 (radius 8) while mean replans fell from 6.2 to 4.7.

## 4. Discussion

**A*'s advantage depends on the heuristic.** A* with a good heuristic does almost no
wasted work on open warehouse layouts, but its advantage depends on how well the heuristic
reflects the map; the detour trap shows the failure mode. Floyd-Warshall solves a harder problem
than a single robot usually needs.

**Exact versus heuristic optimisation.** Exact TSP solvers were fine up to about k = 12 in this
implementation, while the genetic algorithm's runtime was roughly flat in k and its quality
degraded gracefully (about 2% excess at k = 16). Seeding with a good heuristic tour mattered more
as k grew.

**Local navigation was dominated by sensing and by simple design choices.** The largest
improvements for both controllers came from fixing what the robot could sense (rays, walls), not
from the controller logic. My fuzzy controller never matched a few hand-set thresholds in
reliability. Fuzzy control's benefit in this study is a larger safety margin, bought with longer
and less smooth paths; I did not test whether a better-tuned rule base would close the
reliability gap.

**Replanning added modest cost.** Replans expanded 25 to 32 cells on average against 55 for the initial plan (they also start nearer the goal), and the main driver of path stretch was how early the robot could see a blockage.

## 5. Limitations and threats to validity

- Everything is simulated: a kinematic robot, noise-free sensors, circular obstacles, no
  dynamics, no hardware.
- The fuzzy rule base and the reactive thresholds were hand-designed and neither was tuned
  systematically. The fuzzy controller was revised after inspecting failures on development
  worlds; the held-out evaluation limits but does not remove the risk of design choices
  favouring one controller. The decision to move from 3 to 7 sensors came from a diagnosis and
  was applied to both controllers.
- Some statistics are based on few instances (6 to 10 TSP instances per setting, 4 for the large
  GA runs), so differences between settings in those tables are indicative.
- Six paired tests were run without multiple-comparison correction.
- Timings are from pure Python on one machine; scaling trends are meaningful, absolute speeds
  are not.
- In the replanning experiment, blockages are placed next to the initial route and the robot
  senses through shelves, so success rates are not predictions of real-world performance.
- The global planner and the local controller were evaluated separately and not integrated.

## 6. Conclusion and future work

The study produced working implementations and measured trade-offs for each layer of a simple
warehouse-robot navigation stack, and several results contradicted my initial expectations:
branch and bound was slower than dynamic programming in practice, all-pairs precomputation
was rarely worth it, and a threshold controller beat the fuzzy controller on reliability. Natural
next steps are to combine the A* route with the local controller, to diagnose and fix the
fuzzy controller's timeouts, to tune the rule base by optimisation on a training set of worlds
while keeping a held-out set, and to port the planners to a compiled language for realistic
timings.
