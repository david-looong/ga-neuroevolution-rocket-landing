#!/usr/bin/env python3
"""
Showcase the project in two graphs and four headline numbers.

Trains the *baseline* GA configuration (from ``config.py``) ``N`` independent
times with deterministic seeds 1..N, then plots metrics that are averaged
across those seeds (with ±1 σ bands) so no single lucky run can dominate the
story.

Outputs (under ``results/``):

    showcase_data.json                 raw per-trial per-generation metrics
    showcase_headline.json             the four headline numbers
    showcase_fitness_dark.png          best & mean fitness vs generation (dark)
    showcase_fitness_light.png         best & mean fitness vs generation (light)
    showcase_landing_dark.png          landing rate vs generation (dark)
    showcase_landing_light.png         landing rate vs generation (light)

Usage:

    python showcase.py                 # train 5 seeds, plot, print numbers
    python showcase.py --trials 7      # use a different number of seeds
    python showcase.py --plot-only     # re-plot from saved JSON (no training)

Headline numbers reported (mean ± std across seeds):

    1. Neural-network parameters evolved
    2. Final landing success rate (last 10 generations)
    3. Improvement factor: final best fitness / generation-1 best fitness
    4. Rocket landings simulated per training run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import (
    POPULATION_SIZE, NUM_GENERATIONS, TOURNAMENT_SIZE, CROSSOVER_RATE,
    MUTATION_RATE, MUTATION_SIGMA, ELITISM_COUNT, NUM_EVAL_TRIALS,
    NN_LAYERS,
)
from controller import NeuralNetwork
import main as ga_main


# ── Output paths ──────────────────────────────────────────────────────
RESULTS_DIR = "results"
DATA_JSON = os.path.join(RESULTS_DIR, "showcase_data.json")
HEADLINE_JSON = os.path.join(RESULTS_DIR, "showcase_headline.json")
FITNESS_DARK_PNG = os.path.join(RESULTS_DIR, "showcase_fitness_dark.png")
FITNESS_LIGHT_PNG = os.path.join(RESULTS_DIR, "showcase_fitness_light.png")
LANDING_DARK_PNG = os.path.join(RESULTS_DIR, "showcase_landing_dark.png")
LANDING_LIGHT_PNG = os.path.join(RESULTS_DIR, "showcase_landing_light.png")

# Window over which we compute the "final" landing rate (averaging the tail
# trims single-generation noise without smearing across the learning curve).
FINAL_WINDOW = 10


# ═══════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════

def _run_one_trial(seed: int, num_generations: int) -> dict[str, Any]:
    """Run baseline GA once with the given RNG seed."""
    # ``main.train`` reads the seed from ``config.RANDOM_SEED`` at module load
    # time (see ``train`` body). To get reproducible distinct seeds we patch
    # the module attribute before each call.
    ga_main.RANDOM_SEED = int(seed)

    print(f"\n── Trial seed={seed}  "
          f"(pop={POPULATION_SIZE}, gens={num_generations}, "
          f"trials/genome={NUM_EVAL_TRIALS}) ──")

    t0 = time.time()
    result = ga_main.train(
        headless=True,
        run_id=f"showcase_seed{seed}",
        population_size=POPULATION_SIZE,
        num_generations=num_generations,
        tournament_size=TOURNAMENT_SIZE,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        mutation_sigma=MUTATION_SIGMA,
        elitism_count=ELITISM_COUNT,
        num_eval_trials=NUM_EVAL_TRIALS,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s  "
          f"(final best={result['summary']['final_best_fitness']:.1f}, "
          f"final landed={result['summary']['final_landing_rate']:.1%})")
    return {
        "seed": int(seed),
        "elapsed_seconds": round(elapsed, 1),
        "generations": result["generations"],
        "summary": result["summary"],
    }


def train_all_trials(num_trials: int, num_generations: int) -> dict[str, Any]:
    """Train ``num_trials`` independent baseline runs and return raw metrics."""
    trials = []
    for s in range(1, num_trials + 1):
        trials.append(_run_one_trial(seed=s, num_generations=num_generations))

    data = {
        "config": {
            "population_size": POPULATION_SIZE,
            "num_generations": num_generations,
            "tournament_size": TOURNAMENT_SIZE,
            "crossover_rate": CROSSOVER_RATE,
            "mutation_rate": MUTATION_RATE,
            "mutation_sigma": MUTATION_SIGMA,
            "elitism_count": ELITISM_COUNT,
            "num_eval_trials": NUM_EVAL_TRIALS,
            "nn_layers": list(NN_LAYERS),
        },
        "num_trials": int(num_trials),
        "trials": trials,
    }
    return data


def save_data(data: dict[str, Any], path: str = DATA_JSON) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  saved {path}")


def load_data(path: str = DATA_JSON) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
#  Aggregation
# ═══════════════════════════════════════════════════════════════════════

def _stack_metric(data: dict[str, Any], metric: str) -> np.ndarray:
    """Return a (num_trials, num_generations) array of the named metric."""
    arrs = []
    for t in data["trials"]:
        arrs.append([row[metric] for row in t["generations"]])
    return np.asarray(arrs, dtype=float)


def aggregate(data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-generation mean / std across trials for each metric."""
    out: dict[str, np.ndarray] = {}
    for m in ("best_fitness", "mean_fitness", "landing_rate"):
        stack = _stack_metric(data, m)
        out[f"{m}_per_trial"] = stack
        out[f"{m}_mean"] = stack.mean(axis=0)
        out[f"{m}_std"] = stack.std(axis=0)
    out["generation"] = np.asarray(
        [row["generation"] for row in data["trials"][0]["generations"]],
        dtype=int,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Headline numbers
# ═══════════════════════════════════════════════════════════════════════

def compute_headline(data: dict[str, Any]) -> dict[str, Any]:
    cfg = data["config"]
    agg = aggregate(data)

    # 1. NN parameters evolved (a single integer, not stochastic)
    nn_params = NeuralNetwork.genome_size(cfg["nn_layers"])

    # 2. Final landing rate — mean of the last FINAL_WINDOW gens, then mean ± std across trials
    landing = agg["landing_rate_per_trial"]                 # (trials, gens)
    final_landing_per_trial = landing[:, -FINAL_WINDOW:].mean(axis=1)
    final_landing_mean = float(final_landing_per_trial.mean())
    final_landing_std = float(final_landing_per_trial.std())

    # 3. Improvement factor — final-window best fitness / generation-1 best fitness
    best = agg["best_fitness_per_trial"]                    # (trials, gens)
    final_best_per_trial = best[:, -FINAL_WINDOW:].mean(axis=1)
    initial_best_per_trial = best[:, 0]
    safe_initial = np.where(initial_best_per_trial > 0,
                            initial_best_per_trial, np.nan)
    improvement_per_trial = final_best_per_trial / safe_initial
    improvement_mean = float(np.nanmean(improvement_per_trial))
    improvement_std = float(np.nanstd(improvement_per_trial))

    final_best_mean = float(final_best_per_trial.mean())
    final_best_std = float(final_best_per_trial.std())
    initial_best_mean = float(initial_best_per_trial.mean())
    initial_best_std = float(initial_best_per_trial.std())

    # 4. Rocket landings simulated per training run
    landings_per_run = (cfg["population_size"]
                        * cfg["num_generations"]
                        * cfg["num_eval_trials"])

    return {
        "num_trials": data["num_trials"],
        "final_window_generations": FINAL_WINDOW,
        "nn_parameters_evolved": int(nn_params),
        "final_landing_rate_mean": final_landing_mean,
        "final_landing_rate_std": final_landing_std,
        "final_best_fitness_mean": final_best_mean,
        "final_best_fitness_std": final_best_std,
        "initial_best_fitness_mean": initial_best_mean,
        "initial_best_fitness_std": initial_best_std,
        "improvement_factor_mean": improvement_mean,
        "improvement_factor_std": improvement_std,
        "rocket_landings_per_training_run": int(landings_per_run),
    }


def print_headline(h: dict[str, Any]) -> None:
    nt = h["num_trials"]
    win = h["final_window_generations"]
    sep = "─" * 64
    print()
    print(sep)
    print(f"  HEADLINE NUMBERS  (mean ± std across {nt} independent seeds;")
    print(f"  'final' = average of the last {win} generations)")
    print(sep)
    print(f"  Neural-network parameters evolved : {h['nn_parameters_evolved']:>10d}")
    print(f"  Final landing success rate        : "
          f"{h['final_landing_rate_mean']*100:>6.1f}% ± "
          f"{h['final_landing_rate_std']*100:.1f}%")
    print(f"  Final best fitness                : "
          f"{h['final_best_fitness_mean']:>6.1f}  ± "
          f"{h['final_best_fitness_std']:.1f}   "
          f"(gen 1: {h['initial_best_fitness_mean']:.1f} ± "
          f"{h['initial_best_fitness_std']:.1f})")
    print(f"  Improvement factor (final / gen 1): "
          f"{h['improvement_factor_mean']:>6.2f}× ± "
          f"{h['improvement_factor_std']:.2f}×")
    print(f"  Rocket landings simulated per run : "
          f"{h['rocket_landings_per_training_run']:>10,d}")
    print(sep)


def save_headline(h: dict[str, Any], path: str = HEADLINE_JSON) -> None:
    with open(path, "w") as f:
        json.dump(h, f, indent=2)
    print(f"  saved {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

# Hand-picked accent palette: vivid enough for a dark slide, dark enough for a
# light slide. ``best`` / ``mean`` are matched per-theme.
_THEME = {
    "dark": {
        "fg":         "#f2f5ff",
        "muted":      "#a8b0cc",
        "grid":       "#a8b0cc",
        "best_color": "#00e5c8",
        "mean_color": "#ffd042",
        "land_color": "#7aeb7a",
        "legend_bg":  "#1a1d2e",
    },
    "light": {
        "fg":         "#1a1d2e",
        "muted":      "#3a4055",
        "grid":       "#7a8094",
        "best_color": "#0d8c7c",
        "mean_color": "#c08410",
        "land_color": "#2a8b3c",
        "legend_bg":  "#ffffff",
    },
}


def _style_axes(ax: plt.Axes, theme: dict[str, str]) -> None:
    fg = theme["fg"]
    muted = theme["muted"]
    grid = theme["grid"]
    ax.tick_params(axis="both", colors=muted, labelsize=11, width=2, length=6)
    for spine in ax.spines.values():
        spine.set_color(muted)
        spine.set_linewidth(2)
    ax.grid(True, linestyle="--", linewidth=1.2, alpha=0.35, color=grid)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)


def _style_legend(leg, theme: dict[str, str]) -> None:
    leg.get_frame().set_facecolor(theme["legend_bg"])
    leg.get_frame().set_edgecolor(theme["fg"])
    leg.get_frame().set_linewidth(2)
    for text in leg.get_texts():
        text.set_color(theme["fg"])
        text.set_fontweight("bold")


def _save_transparent(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_fitness(data: dict[str, Any], theme_name: str, out_path: str) -> None:
    theme = _THEME[theme_name]
    agg = aggregate(data)
    gens = agg["generation"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    best_mean = agg["best_fitness_mean"]
    best_std = agg["best_fitness_std"]
    mean_mean = agg["mean_fitness_mean"]
    mean_std = agg["mean_fitness_std"]

    ax.fill_between(gens, mean_mean - mean_std, mean_mean + mean_std,
                    color=theme["mean_color"], alpha=0.18, linewidth=0)
    ax.plot(gens, mean_mean, color=theme["mean_color"],
            linewidth=2.6, solid_capstyle="round",
            label="Mean population fitness")

    ax.fill_between(gens, best_mean - best_std, best_mean + best_std,
                    color=theme["best_color"], alpha=0.20, linewidth=0)
    ax.plot(gens, best_mean, color=theme["best_color"],
            linewidth=3.2, solid_capstyle="round",
            label="Best individual fitness")

    ax.set_xlabel("Generation", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Fitness (mean ± std across {data['num_trials']} seeds)",
                  fontsize=13, fontweight="bold")
    ax.set_title("Evolution learns: fitness over generations",
                 fontsize=15, fontweight="bold", pad=14)

    _style_axes(ax, theme)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=4))
    ax.set_xlim(gens.min(), gens.max())

    leg = ax.legend(loc="lower right", fontsize=11, framealpha=0.92)
    _style_legend(leg, theme)

    _save_transparent(fig, out_path)


def plot_landing(data: dict[str, Any], theme_name: str, out_path: str) -> None:
    theme = _THEME[theme_name]
    agg = aggregate(data)
    gens = agg["generation"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    rate_mean = agg["landing_rate_mean"] * 100
    rate_std = agg["landing_rate_std"] * 100

    ax.fill_between(gens, rate_mean - rate_std, rate_mean + rate_std,
                    color=theme["land_color"], alpha=0.22, linewidth=0)
    ax.plot(gens, rate_mean, color=theme["land_color"],
            linewidth=3.2, solid_capstyle="round",
            label="Landing success rate")

    ax.set_xlabel("Generation", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Rockets landed (mean ± std across "
                  f"{data['num_trials']} seeds)",
                  fontsize=13, fontweight="bold")
    ax.set_title("Rockets actually land: success rate over generations",
                 fontsize=15, fontweight="bold", pad=14)

    _style_axes(ax, theme)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=4))
    ax.set_xlim(gens.min(), gens.max())
    ax.set_ylim(-2, 102)

    leg = ax.legend(loc="lower right", fontsize=11, framealpha=0.92)
    _style_legend(leg, theme)

    _save_transparent(fig, out_path)


def plot_all(data: dict[str, Any]) -> None:
    print("\nPlotting…")
    plot_fitness(data, "dark", FITNESS_DARK_PNG)
    plot_fitness(data, "light", FITNESS_LIGHT_PNG)
    plot_landing(data, "dark", LANDING_DARK_PNG)
    plot_landing(data, "light", LANDING_LIGHT_PNG)


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate showcase graphs and headline numbers.")
    ap.add_argument("--trials", type=int, default=5,
                    help="number of independent seeds to train (default: 5)")
    ap.add_argument("--generations", type=int, default=NUM_GENERATIONS,
                    help=f"generations per trial (default: {NUM_GENERATIONS})")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip training; re-plot from saved JSON")
    ap.add_argument("--data", default=DATA_JSON,
                    help=f"raw-data JSON path (default: {DATA_JSON})")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.plot_only:
        if not os.path.exists(args.data):
            raise SystemExit(f"--plot-only set but {args.data} does not exist; "
                             "run without --plot-only first.")
        data = load_data(args.data)
        print(f"Loaded {args.data}  ({data['num_trials']} trials, "
              f"{len(data['trials'][0]['generations'])} generations)")
    else:
        data = train_all_trials(num_trials=args.trials,
                                num_generations=args.generations)
        save_data(data, args.data)

    headline = compute_headline(data)
    save_headline(headline)
    print_headline(headline)
    plot_all(data)
    print()


if __name__ == "__main__":
    main()
