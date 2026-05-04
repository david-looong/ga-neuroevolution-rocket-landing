"""
All hyperparameters for the 2D Rocket Landing GA.
Edit this file to tune the simulation, controller, GA, novelty, and visualization.
"""
import numpy as np

# ========================= RANDOM SEED =========================
RANDOM_SEED = 42  # Set to None for non-reproducible runs

# ========================= PHYSICS ==============================
GRAVITY = 10.0              # m/s² (base value, subject to curriculum randomization)
MOMENT_OF_INERTIA = 100.0   # kg·m²
MAX_THRUST = 1500.0         # N (base; gives thrust-to-weight ≈ 2.55)
MAX_GIMBAL_ANGLE = np.radians(15)  # ±15° gimbal range
THRUSTER_ARM = 3.0          # m, distance from center of gravity to nozzle
INITIAL_FUEL = 1.0          # normalized [0, 1]
FUEL_CONSUMPTION = 0.10     # normalized fuel/s at full throttle (~10 s burn)
MIN_THROTTLE = 0.4
ENGINE_OFF_THRESHOLD = 0.20  # below this, the single-use engine shuts down
ENGINE_IGNITION_THRESHOLD = 0.50  # first ignition must be deliberate
DRY_MASS = 60.0             # kg (structure + engines)
FUEL_MASS = 40.0            # kg at fuel=1.0

# Rocket geometry
ROCKET_SCALE = 1.25         # global rocket size multiplier (hitbox)
ROCKET_BASE_HEIGHT = 12.0       # m, baseline full rocket height at scale=1
ROCKET_HEIGHT = ROCKET_BASE_HEIGHT * ROCKET_SCALE     # m, full rocket height
ROCKET_HITBOX_WIDTH = ROCKET_HEIGHT * (4.0 / 12.0)    # m, 4x12 pixel-art body
ROCKET_HITBOX_HEIGHT = ROCKET_HEIGHT                  # m, full image height

# Atmospheric drag: F_drag = -0.5 · ρ · |v| · v · Cd_A
AIR_DENSITY = 1.225         # kg/m³
DRAG_CD_A = 0.7             # Cd × reference area (m²), tunable

# Simulation timing
SIM_DT = 1.0 / 30.0        # 60 Hz physics timestep
MAX_SIM_TIME = 30.0         # seconds per episode

# Landing pad
PAD_X = 0.0                 # pad center (m)
PAD_WIDTH = 40.0            # total pad width (m)

# ========================= WIND (Ornstein–Uhlenbeck) ============
WIND_OU_THETA = 0.3         # mean-reversion rate

# ========================= INITIAL CONDITIONS ===================
ALTITUDE_RANGE = (400.0, 600.0)
HORIZONTAL_RANGE = (-250.0, 250.0)
INIT_VY_RANGE = (-80.0, -20.0)      # downward
INIT_VX_RANGE = (-40.0, 40.0)
INIT_ANGLE_RANGE = np.radians(60)   # ±20°

# ALTITUDE_RANGE = (400.0, 500.0)
# HORIZONTAL_RANGE = (-200.0, 200.0)
# INIT_VY_RANGE = (-100.0, -20.0)      # downward
# INIT_VX_RANGE = (-50.0, 50.0)
# INIT_ANGLE_RANGE = np.radians(60)   # ±20°

# ========================= CURRICULUM SCHEDULE ==================
# (start_gen, gravity_variation, thrust_variation, max_wind_m/s)
#
# The schedule lets the population learn basic control in a calm, fixed world
# before introducing perturbations that demand robust, adaptive strategies.
CURRICULUM = [
    (0,   0.00, 0.00,  0.0),   # Gen 0–29:   fixed physics, no wind
    (30,  0.10, 0.10,  5.0),   # Gen 30–99:  ±10 % variation, light gusts
    (100, 0.20, 0.20, 15.0),   # Gen 100+:   full randomization
]

NUM_EVAL_TRIALS = 8  # scenarios per genome (averaged for noise reduction)

# ========================= NEURAL NETWORK =======================
NN_LAYERS = [6, 16, 16, 2]  # input → hidden → hidden → output

# Input normalization divisors (keep inputs roughly in [-1, 1])
NORM_X = 200.0
NORM_Y = 400.0
NORM_VX = 80.0
NORM_VY = 80.0
NORM_THETA = np.pi
NORM_OMEGA = 5.0

# ========================= GENETIC ALGORITHM ====================
POPULATION_SIZE = 100
NUM_GENERATIONS = 300
TOURNAMENT_SIZE = 7
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1       # per-gene probability
MUTATION_SIGMA = 0.1        # Gaussian noise σ
ELITISM_COUNT = 5

# ========================= NOVELTY SEARCH =======================
# Novelty bonus encourages behavioural diversity so the population explores
# many landing strategies instead of collapsing to a single policy.
# Tune NOVELTY_WEIGHT:  too high → chases weirdness, too low → one strategy.
NOVELTY_K = 15              # k nearest neighbors
NOVELTY_WEIGHT = 15.0       # bonus weight (scaled so max bonus ≈ this value)
NOVELTY_ARCHIVE_PROB = 0.05 # probability of archiving each behaviour

# ========================= VISUALIZATION ========================
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 800
VIS_SCALE = 2.4             # pixels per meter
VIS_FPS = 60
