#!/usr/bin/env python3
"""
2D Rocket Landing via Genetic Algorithm with Novelty Search.

Run:  python main.py

After each generation all genomes are replayed simultaneously in a pygame
window (best in full colour, others semi-transparent).
  [SPACE]  skip to the next generation
  [N]      cycle through behaviourally diverse genomes from the same generation
  Close the window to continue training headless.

Training of the next generation proceeds in a background process while the
preview is playing, so there's no idle time between generations.

At the end of training a fitness-history plot is saved to fitness_history.png.
"""

import time
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from threading import Thread

from config import (
    RANDOM_SEED, GRAVITY, ROCKET_MASS, MOMENT_OF_INERTIA, MAX_THRUST,
    MAX_GIMBAL_ANGLE, THRUSTER_ARM, FUEL_CONSUMPTION, AIR_DENSITY, DRAG_CD_A,
    SIM_DT, MAX_SIM_TIME, PAD_X, PAD_WIDTH, WIND_OU_THETA, ALTITUDE_RANGE,
    HORIZONTAL_RANGE, INIT_VY_RANGE, INIT_VX_RANGE, INIT_ANGLE_RANGE,
    INITIAL_FUEL, CURRICULUM, NUM_EVAL_TRIALS, NN_LAYERS, NORM_X, NORM_Y,
    NORM_VX, NORM_VY, NORM_THETA, NORM_OMEGA, POPULATION_SIZE, NUM_GENERATIONS,
    TOURNAMENT_SIZE, CROSSOVER_RATE, MUTATION_RATE, MUTATION_SIGMA,
    ELITISM_COUNT, NOVELTY_K, NOVELTY_WEIGHT, NOVELTY_ARCHIVE_PROB,
    SCREEN_WIDTH, SCREEN_HEIGHT, VIS_SCALE, VIS_FPS,
)
from physics import RocketState, RocketSim, WindModel
from controller import NeuralNetwork
from ga import GeneticAlgorithm
from novelty import NoveltyArchive


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def get_curriculum(generation: int) -> dict:
    """Return domain-randomisation parameters for the current generation.

    The curriculum ramps up difficulty so the population can learn basic
    flight control in a calm world before wind and physics variation are
    introduced.
    """
    active = CURRICULUM[0]
    for entry in CURRICULUM:
        if generation >= entry[0]:
            active = entry
    return {"gravity_var": active[1], "thrust_var": active[2],
            "max_wind": active[3]}


def make_sim(cur: dict, rng: np.random.Generator):
    """Build a RocketSim with curriculum-randomised physics."""
    gv, tv, mw = cur["gravity_var"], cur["thrust_var"], cur["max_wind"]
    g = GRAVITY * (1 + rng.uniform(-gv, gv)) if gv else GRAVITY
    t = MAX_THRUST * (1 + rng.uniform(-tv, tv)) if tv else MAX_THRUST
    wind = WindModel(ou_theta=WIND_OU_THETA, max_wind=mw, rng=rng)
    return RocketSim(
        gravity=g, mass=ROCKET_MASS, moi=MOMENT_OF_INERTIA,
        max_thrust=t, max_gimbal=MAX_GIMBAL_ANGLE,
        thruster_arm=THRUSTER_ARM, fuel_rate=FUEL_CONSUMPTION,
        air_density=AIR_DENSITY, drag_cd_a=DRAG_CD_A,
        dt=SIM_DT, max_time=MAX_SIM_TIME,
        pad_x=PAD_X, pad_half_width=PAD_WIDTH / 2,
        wind_model=wind,
    )


def random_initial(rng: np.random.Generator) -> RocketState:
    return RocketState(
        x=rng.uniform(*HORIZONTAL_RANGE),
        y=rng.uniform(*ALTITUDE_RANGE),
        vx=rng.uniform(*INIT_VX_RANGE),
        vy=rng.uniform(*INIT_VY_RANGE),
        theta=rng.uniform(-INIT_ANGLE_RANGE, INIT_ANGLE_RANGE),
        omega=rng.uniform(-0.1, 0.1),
        fuel=INITIAL_FUEL,
    )


def normalize(state: RocketState) -> np.ndarray:
    return np.array([
        (state.x - PAD_X) / NORM_X,
        state.y / NORM_Y,
        state.vx / NORM_VX,
        state.vy / NORM_VY,
        state.theta / NORM_THETA,
        state.omega / NORM_OMEGA,
    ])


# ═══════════════════════════════════════════════════════════════════
#  Fitness  (staged / gated)
# ═══════════════════════════════════════════════════════════════════

def compute_fitness(sim: RocketSim) -> float:
    """
    Staged fitness avoids the "do nothing" local optimum.

    1. Proximity to pad   (40 pts) — always; √-scaled for a gentle gradient
    2. Descent to ground  (20 pts) — always
    3. Low terminal speed (25 pts) — gated on being near the ground
    4. Upright at end     (15 pts) — gated on being near the ground
    5. Soft impact        (20 pts) — gated on ground contact; rewards low
                                     impact speed at the moment of touchdown
    6. Survival time       (5 pts) — small incentive for staying alive
    7. Fuel efficiency    (40 pts) — ONLY awarded on a successful landing

    Max ≈ 165.  Early genomes score mostly on (1) and (2), giving them a
    gradient to climb even before they manage soft landings.
    """
    s = sim.state
    dx = abs(s.x - sim.pad_x)
    speed = np.sqrt(s.vx ** 2 + s.vy ** 2)

    proximity = max(0.0, 1.0 - (dx / 200.0) ** 0.5) * 40.0
    descent   = max(0.0, 1.0 - s.y / 300.0) * 20.0

    # Gate speed / tilt on actually reaching the ground neighbourhood.
    # on_ground gives full credit; otherwise ramp up as altitude drops below 50 m.
    ground_gate = 1.0 if sim.on_ground else max(0.0, 1.0 - s.y / 50.0)
    speed_pts   = max(0.0, 1.0 - speed / 40.0) * ground_gate * 25.0
    tilt_pts    = max(0.0, 1.0 - abs(s.theta) / (np.pi / 2)) * ground_gate * 15.0

    # Reward soft touchdowns — impact_speed is recorded at first ground contact.
    # Full 20 pts at 0 m/s impact, 0 pts at ≥30 m/s. Only applies if the
    # rocket actually reached the ground.
    if sim.on_ground:
        impact_pts = max(0.0, 1.0 - sim.impact_speed / 30.0) * 20.0
    else:
        impact_pts = 0.0

    survival    = (sim.time / sim.max_time) * 5.0
    fuel_bonus  = s.fuel * 40.0 if sim.landed else 0.0

    return (proximity + descent + speed_pts + tilt_pts
            + impact_pts + survival + fuel_bonus)


# ═══════════════════════════════════════════════════════════════════
#  Behaviour descriptor  (for novelty)
# ═══════════════════════════════════════════════════════════════════

class BehaviourAccumulator:
    """Lightweight online accumulator — avoids storing full trajectories."""

    __slots__ = ("alt_sum", "max_tilt", "horiz_dist", "prev_x", "n")

    def __init__(self, state: RocketState):
        self.alt_sum = state.y
        self.max_tilt = abs(state.theta)
        self.horiz_dist = 0.0
        self.prev_x = state.x
        self.n = 1

    def update(self, state: RocketState):
        self.alt_sum += state.y
        self.n += 1
        t = abs(state.theta)
        if t > self.max_tilt:
            self.max_tilt = t
        self.horiz_dist += abs(state.x - self.prev_x)
        self.prev_x = state.x

    def descriptor(self, sim: RocketSim) -> np.ndarray:
        return np.array([
            self.alt_sum / self.n,        # average altitude
            self.max_tilt,                 # max absolute tilt angle
            1.0 - sim.state.fuel,          # fuel consumed
            sim.time,                      # time to ground / timeout
            self.horiz_dist,               # total horizontal distance
        ])


# ═══════════════════════════════════════════════════════════════════
#  Evaluation  (one genome — used by worker processes)
# ═══════════════════════════════════════════════════════════════════

def _evaluate_one(genome: np.ndarray, cur: dict, trial_seeds: list[int]):
    """Evaluate one genome over multiple randomised scenarios.

    Designed to run in a worker process — creates its own NN / Sim instances.
    Returns (fitness, behaviour_vec, landed_any).
    """
    nn = NeuralNetwork(NN_LAYERS)
    nn.set_genome(genome)
    fits, bvecs = [], []
    landed_any = False

    for seed in trial_seeds:
        trng = np.random.default_rng(seed)
        sim = make_sim(cur, trng)
        initial = random_initial(trng)
        sim.reset(initial)
        acc = BehaviourAccumulator(sim.state)

        while True:
            inp = normalize(sim.state)
            throttle, gimbal = nn.forward(inp)
            done, _info = sim.step(throttle, gimbal)
            acc.update(sim.state)
            if done:
                break

        fits.append(compute_fitness(sim))
        bvecs.append(acc.descriptor(sim))
        if sim.landed:
            landed_any = True

    return float(np.mean(fits)), np.mean(bvecs, axis=0), landed_any


def _worker_batch(args):
    """Evaluate a batch of genomes (reduces IPC overhead vs one-per-call)."""
    genomes, cur, trial_seeds = args
    results = []
    for g in genomes:
        results.append(_evaluate_one(g, cur, trial_seeds))
    return results


# ═══════════════════════════════════════════════════════════════════
#  Diverse genome selection  (for the [N] key in the viewer)
# ═══════════════════════════════════════════════════════════════════

def pick_diverse(population, behaviours, fitnesses, n=5):
    """Greedily pick *n* genomes that are far apart in behaviour space."""
    if len(population) <= n:
        return list(range(len(population)))

    mean = behaviours.mean(axis=0)
    std  = behaviours.std(axis=0) + 1e-8
    normed = (behaviours - mean) / std

    selected = [int(np.argmax(fitnesses))]
    for _ in range(n - 1):
        min_d = np.full(len(population), np.inf)
        for si in selected:
            d = np.sqrt(np.sum((normed - normed[si]) ** 2, axis=1))
            min_d = np.minimum(min_d, d)
        for si in selected:
            min_d[si] = -1.0
        selected.append(int(np.argmax(min_d)))
    return selected


# ═══════════════════════════════════════════════════════════════════
#  Parallel evaluation helper
# ═══════════════════════════════════════════════════════════════════

def evaluate_population(population, cur, trial_seeds, pool):
    """Evaluate all genomes in parallel using the process pool."""
    import os
    n_workers = os.cpu_count() or 4
    batch_size = max(1, len(population) // n_workers)
    batches = []
    for i in range(0, len(population), batch_size):
        chunk = population[i:i + batch_size]
        batches.append((chunk, cur, trial_seeds))

    fitnesses  = np.zeros(len(population))
    behaviours = np.zeros((len(population), 5))
    landed     = np.zeros(len(population), dtype=bool)

    idx = 0
    for batch_results in pool.map(_worker_batch, batches):
        for f, b, l in batch_results:
            fitnesses[idx]  = f
            behaviours[idx] = b
            landed[idx]     = l
            idx += 1

    return fitnesses, behaviours, landed


# ═══════════════════════════════════════════════════════════════════
#  Core training loop  (callable from experiment.py or interactively)
# ═══════════════════════════════════════════════════════════════════

def train(
    headless: bool = True,
    run_id: str = "default",
    population_size: int = POPULATION_SIZE,
    num_generations: int = NUM_GENERATIONS,
    tournament_size: int = TOURNAMENT_SIZE,
    crossover_rate: float = CROSSOVER_RATE,
    mutation_rate: float = MUTATION_RATE,
    mutation_sigma: float = MUTATION_SIGMA,
    elitism_count: int = ELITISM_COUNT,
) -> dict:
    """Run the GA training loop and return per-generation metrics.

    Returns:
        {
            "generations": list of per-generation metric dicts,
            "summary":     single dict with run-level summary stats,
        }
    """
    rng = np.random.default_rng(RANDOM_SEED)
    genome_size = NeuralNetwork.genome_size(NN_LAYERS)

    import os
    n_workers = os.cpu_count() or 4

    if not headless:
        print(f"Genome size : {genome_size} parameters")
        print(f"Population  : {population_size}   Generations: {num_generations}")
        print(f"Trials/genome: {NUM_EVAL_TRIALS}   Workers: {n_workers}\n")

    ga = GeneticAlgorithm(
        pop_size=population_size, genome_size=genome_size,
        tournament_size=tournament_size, crossover_rate=crossover_rate,
        mutation_rate=mutation_rate, mutation_sigma=mutation_sigma,
        elitism_count=elitism_count,
    )
    archive = NoveltyArchive(k=NOVELTY_K, archive_prob=NOVELTY_ARCHIVE_PROB)
    population = ga.initialize(rng)

    gen_metrics: list[dict] = []
    best_hist: list[float] = []
    mean_hist: list[float] = []
    renderer = None

    bg_result = None
    bg_thread = None
    run_start = time.time()

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:

            for gen in range(num_generations):
                cur = get_curriculum(gen)
                trial_seeds = [int(rng.integers(0, 2 ** 31))
                               for _ in range(NUM_EVAL_TRIALS)]

                # ── Use background results if available, else evaluate now ──
                if bg_result is not None:
                    fitnesses, behaviours, landed, population, cur_used, elapsed = bg_result
                    bg_result = None
                else:
                    t0 = time.time()
                    fitnesses, behaviours, landed = evaluate_population(
                        population, cur, trial_seeds, pool)
                    elapsed = time.time() - t0
                    cur_used = cur

                # ── novelty bonus ────────────────────────────────────
                nov = archive.compute_novelty(behaviours)
                nmax = nov.max()
                nov_norm = nov / nmax if nmax > 0 else nov
                combined = fitnesses + NOVELTY_WEIGHT * nov_norm
                archive.update(behaviours, rng)

                # ── stats ────────────────────────────────────────────
                best_idx = int(np.argmax(combined))
                best_fit = fitnesses[best_idx]
                mean_fit = float(fitnesses.mean())
                suc_rate = float(landed.mean())

                best_hist.append(best_fit)
                mean_hist.append(mean_fit)

                phase = ("easy" if cur_used["max_wind"] == 0
                         else "medium" if cur_used["max_wind"] <= 5 else "hard")

                gen_metrics.append({
                    "run_id": run_id,
                    "generation": gen + 1,
                    "best_fitness": round(best_fit, 4),
                    "mean_fitness": round(mean_fit, 4),
                    "landing_rate": round(suc_rate, 4),
                    "archive_size": len(archive.archive),
                    "phase": phase,
                    "elapsed_seconds": round(elapsed, 3),
                })

                print(f"Gen {gen+1:3d}/{num_generations} │ "
                      f"Best {best_fit:6.1f}  Mean {mean_fit:5.1f} │ "
                      f"Landed {suc_rate:5.1%} │ "
                      f"Archive {len(archive.archive):4d} │ "
                      f"{phase:6s} │ {elapsed:.1f}s")
                sys.stdout.flush()

                # ── breed next generation ────────────────────────────
                next_pop = ga.next_generation(population, combined, rng)

                # ── kick off background eval for next gen while preview plays ──
                next_cur = get_curriculum(gen + 1) if gen + 1 < num_generations else None
                if next_cur is not None:
                    next_seeds = [int(rng.integers(0, 2 ** 31))
                                  for _ in range(NUM_EVAL_TRIALS)]

                    def _bg_eval(pop=next_pop, c=next_cur, seeds=next_seeds):
                        t0 = time.time()
                        f, b, l = evaluate_population(pop, c, seeds, pool)
                        return (f, b, l, pop, c, time.time() - t0)

                    def _run_bg(fn):
                        nonlocal bg_result
                        bg_result = fn()

                    bg_thread = Thread(target=_run_bg,
                                       args=(_bg_eval,), daemon=True)
                    bg_thread.start()

                # ── visualise all genomes (interactive mode only) ─────
                if not headless:
                    try:
                        if renderer is None:
                            from renderer import Renderer
                            renderer = Renderer(SCREEN_WIDTH, SCREEN_HEIGHT,
                                                VIS_SCALE, VIS_FPS)

                        replay_seed = (RANDOM_SEED + gen
                                       if RANDOM_SEED is not None else None)

                        def _make_replay():
                            r = np.random.default_rng(replay_seed)
                            s = make_sim(cur_used, r)
                            i = random_initial(r)
                            return s, i

                        n_show = min(30, population_size)
                        sorted_idxs = np.argsort(combined)[::-1]
                        step = max(1, len(sorted_idxs) // n_show)
                        show_idxs = sorted_idxs[::step][:n_show]
                        if best_idx not in show_idxs:
                            show_idxs = np.concatenate([[best_idx], show_idxs])

                        all_genomes = [(i, population[i]) for i in show_idxs]

                        div_idxs = pick_diverse(population, behaviours, fitnesses, n=5)
                        diverse = [
                            (f"Diverse #{j+1}  fit={fitnesses[di]:.1f}", population[di])
                            for j, di in enumerate(div_idxs)
                        ]

                        gi = {"gen": gen + 1, "total_gen": num_generations,
                              "fitness": best_fit, "success_rate": suc_rate}

                        result = renderer.replay_generation(
                            best_idx, all_genomes, NN_LAYERS, _make_replay,
                            gi, normalize, diverse)

                        if result == "quit":
                            print("  (window closed — continuing headless)")
                            renderer = None

                    except Exception as exc:
                        if renderer is not None:
                            try:
                                renderer.close()
                            except Exception:
                                pass
                            renderer = None
                        print(f"  (vis error: {exc})")

                # ── wait for background eval if still running ─────────
                if bg_thread is not None:
                    bg_thread.join()
                    bg_thread = None

                population = next_pop

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    # ── build summary ────────────────────────────────────────────
    converged_gen = -1
    for m in gen_metrics:
        if m["landing_rate"] >= 0.5:
            converged_gen = m["generation"]
            break

    final = gen_metrics[-1] if gen_metrics else {}
    summary = {
        "run_id": run_id,
        "pop_size": population_size,
        "tournament_size": tournament_size,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "mutation_sigma": mutation_sigma,
        "elitism_count": elitism_count,
        "converged_gen": converged_gen,
        "final_best_fitness": final.get("best_fitness", 0.0),
        "final_mean_fitness": final.get("mean_fitness", 0.0),
        "final_landing_rate": final.get("landing_rate", 0.0),
        "total_seconds": round(time.time() - run_start, 1),
    }

    # ── final fitness plot (interactive mode only) ───────────────
    if not headless and best_hist:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(best_hist, label="Best fitness", linewidth=1.5)
            ax.plot(mean_hist, label="Mean fitness", alpha=0.7)
            ax.set_xticks(range(0, num_generations + 1, 25))
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness")
            ax.set_title("Fitness over generations")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig("fitness_history.png", dpi=150)
            print("\nFitness plot saved to fitness_history.png")
            plt.show()
        except Exception as exc:
            print(f"Could not plot: {exc}")

    return {"generations": gen_metrics, "summary": summary}


def main():
    train(headless=True)


if __name__ == "__main__":
    main()
