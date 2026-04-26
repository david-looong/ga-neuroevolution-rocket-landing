#!/usr/bin/env python3
"""
Hyperparameter sweep for the GA neuroevolution rocket landing system.

Performs a one-factor-at-a-time sweep over the six GA hyperparameters
described in the research question, running multiple trials per configuration
and writing all results to CSV files in the results/ directory.

Usage:
    python experiment.py

Outputs:
    results/runs.csv        — one row per trial (summary statistics)
    results/generations.csv — one row per generation per trial
"""

import csv
import os
import sys

from main import train
from config import (
    POPULATION_SIZE, NUM_GENERATIONS, TOURNAMENT_SIZE,
    CROSSOVER_RATE, MUTATION_RATE, MUTATION_SIGMA, ELITISM_COUNT,
)

# ── Output paths ─────────────────────────────────────────────────
RESULTS_DIR = "results"
RUNS_CSV    = os.path.join(RESULTS_DIR, "runs.csv")
GENS_CSV    = os.path.join(RESULTS_DIR, "generations.csv")

RUNS_FIELDS = [
    "run_id", "pop_size", "tournament_size", "crossover_rate",
    "mutation_rate", "mutation_sigma", "elitism_count", "trial",
    "converged_gen", "final_best_fitness", "final_mean_fitness",
    "final_landing_rate", "total_seconds",
]

GENS_FIELDS = [
    "run_id", "generation", "best_fitness", "mean_fitness",
    "landing_rate", "archive_size", "phase", "elapsed_seconds",
]

# ── Experiment definitions ────────────────────────────────────────
# Baseline values; each experiment varies exactly one parameter.
BASELINE = dict(
    population_size = POPULATION_SIZE,    # 100
    tournament_size = TOURNAMENT_SIZE,    # 5
    crossover_rate  = CROSSOVER_RATE,     # 0.7
    mutation_rate   = MUTATION_RATE,      # 0.1
    mutation_sigma  = MUTATION_SIGMA,     # 0.1
    elitism_count   = ELITISM_COUNT,      # 5
)

# Each entry is a partial override dict (merged with BASELINE).
# The baseline itself appears once (shared across all parameter groups).
PARAM_SWEEPS = [
    # population_size
    {"population_size": 50},
    {"population_size": 100},   # baseline
    {"population_size": 200},
    # tournament_size
    {"tournament_size": 3},
    {"tournament_size": 5},     # baseline (already counted above)
    {"tournament_size": 10},
    # crossover_rate
    {"crossover_rate": 0.5},
    {"crossover_rate": 0.7},    # baseline
    {"crossover_rate": 0.9},
    # mutation_rate
    {"mutation_rate": 0.05},
    {"mutation_rate": 0.1},     # baseline
    {"mutation_rate": 0.2},
    # mutation_sigma
    {"mutation_sigma": 0.05},
    {"mutation_sigma": 0.1},    # baseline
    {"mutation_sigma": 0.2},
    # elitism_count
    {"elitism_count": 1},
    {"elitism_count": 5},       # baseline
    {"elitism_count": 10},
]

NUM_TRIALS = 3


def _build_configs() -> list[dict]:
    """Merge each sweep override with BASELINE and deduplicate."""
    seen = set()
    configs = []
    for override in PARAM_SWEEPS:
        cfg = {**BASELINE, **override}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def _run_id(cfg: dict, trial: int) -> str:
    return (
        f"pop{cfg['population_size']}"
        f"_t{cfg['tournament_size']}"
        f"_cr{cfg['crossover_rate']}"
        f"_mr{cfg['mutation_rate']}"
        f"_s{cfg['mutation_sigma']}"
        f"_e{cfg['elitism_count']}"
        f"_trial{trial}"
    )


def run_sweep(num_generations: int = NUM_GENERATIONS) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    configs = _build_configs()
    total_runs = len(configs) * NUM_TRIALS
    print(f"Experiment sweep: {len(configs)} configs × {NUM_TRIALS} trials = {total_runs} runs")
    print(f"Generations per run: {num_generations}")
    print(f"Results → {RUNS_CSV}  /  {GENS_CSV}\n")

    runs_file = open(RUNS_CSV, "w", newline="")
    gens_file = open(GENS_CSV, "w", newline="")
    runs_writer = csv.DictWriter(runs_file, fieldnames=RUNS_FIELDS)
    gens_writer = csv.DictWriter(gens_file, fieldnames=GENS_FIELDS)
    runs_writer.writeheader()
    gens_writer.writeheader()

    run_num = 0
    try:
        for cfg in configs:
            for trial in range(1, NUM_TRIALS + 1):
                run_num += 1
                rid = _run_id(cfg, trial)
                print(f"\n{'='*60}")
                print(f"Run {run_num}/{total_runs}  id={rid}")
                print(f"{'='*60}")
                sys.stdout.flush()

                result = train(
                    headless=True,
                    run_id=rid,
                    num_generations=num_generations,
                    **cfg,
                )

                # Write per-generation rows immediately (flush after each run)
                for row in result["generations"]:
                    gens_writer.writerow(row)
                gens_file.flush()

                # Write summary row
                summary = result["summary"]
                summary["trial"] = trial
                runs_writer.writerow({k: summary[k] for k in RUNS_FIELDS})
                runs_file.flush()

                print(f"  converged_gen={summary['converged_gen']}  "
                      f"final_best={summary['final_best_fitness']:.1f}  "
                      f"final_landed={summary['final_landing_rate']:.1%}  "
                      f"time={summary['total_seconds']:.0f}s")

    finally:
        runs_file.close()
        gens_file.close()

    print(f"\nDone. {run_num} runs written.")
    print(f"  {RUNS_CSV}  ({run_num} rows)")
    print(f"  {GENS_CSV}  ({run_num * num_generations} rows)")


if __name__ == "__main__":
    # Allow passing a reduced generation count for quick tests:
    #   python experiment.py 50
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_GENERATIONS
    run_sweep(num_generations=gens)
