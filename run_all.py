"""Regenerate every table and figure used in the report, in order. Stops at the first failure."""
import subprocess
import sys
import time

STEPS = [
    ["-m", "pytest", "-q"],
    ["-m", "experiments.run_experiments"],   # Dijkstra vs A*: scaling, density, detour
    ["-m", "experiments.run_apsp"],          # Floyd-Warshall break-even
    ["-m", "experiments.run_tsp"],           # exact TSP solvers
    ["-m", "experiments.run_ga"],            # genetic algorithm
    ["-m", "experiments.run_final_fuzzy"],   # final fuzzy vs reactive comparison
    ["-m", "experiments.run_replanning"],    # replanning under hidden obstacles
    ["-m", "experiments.make_replay"],       # replay GIF
]

if __name__ == "__main__":
    for args in STEPS:
        print(f"\n>>> python {' '.join(args)}", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, *args])
        if result.returncode != 0:
            print(f"FAILED: python {' '.join(args)}")
            sys.exit(result.returncode)
        print(f"<<< done in {time.time() - t0:.0f} s", flush=True)
    print("\nAll experiments regenerated. See ./results/")
