from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pygame

@dataclass
class RocketState:
    x: float        # horizontal position (m)
    y: float        # vertical position (m)
    vx: float       # horizontal velocity (m/s)
    vy: float       # vertical velocity (m/s)
    theta: float    # tilt angle from vertical (radians)
    omega: float    # angular velocity (rad/s)

# Physical constants - can be tuned
GRAVITY     = 9.8   # m/s^2
MASS        = 10.0  # kg
I_MOMENT    = 5.0   # moment of inertia (kg*m^2)
DRAG_COEFF  = 0.01  # linear drag
MAX_THRUST  = 200.0 # Newtons
MAX_GIMBAL  = 0.3   # radians (~17 degrees)
DT          = 0.05  # timestep in seconds

# Pygame constants
SCREEN_W, SCREEN_H = 800, 600
SCALE = 1.5 # pixels per meter, adjust as needed

def step(state: RocketState, thrust: float, gimbal: float) -> RocketState:
    # Clamp controls to physical limits
    thrust = float(np.clip(thrust, 0, MAX_THRUST))
    gimbal = float(np.clip(gimbal, -MAX_GIMBAL, MAX_GIMBAL))

    # Net angle the thrust force acts in world frame
    fire_angle = state.theta + gimbal

    # Acceleration from thrust (in x and y)
    ax = (thrust / MASS) * np.sin(fire_angle)
    ay = (thrust / MASS) * np.cos(fire_angle) - GRAVITY

    # Simple linear drag opposing velocity
    ax -= DRAG_COEFF * state.vx
    ay -= DRAG_COEFF * state.vy

    # Torque from gimbal offset, producing angular acceleration
    alpha = -(thrust * np.sin(gimbal)) / I_MOMENT

    # Euler integration: new = old + rate * dt
    return RocketState(
        x       = float(state.x + state.vx * DT),
        y       = float(state.y + state.vy * DT),
        vx      = float(state.vx + ax * DT),
        vy      = float(state.vy + ay * DT),
        theta   = float(state.theta + state.omega * DT),
        omega   = float(state.omega + alpha * DT),
    )

# Check for termination conditions: landed, crashed, or out of bounds
def check_termination(state: RocketState) -> str | None:
    if state.y <= 0:
        v_mag = np.sqrt(state.vx**2 + state.vy**2)
        if v_mag < 5.0 and abs(state.theta) < 0.2:
            return "landed"
        else:
            return "crashed"
    if abs(state.x) > 200 or state.y > 500:
        return "out_of_bounds"
    return None # episode still running

# Run a full episode with given controls, returning the trajectory history for plotting
def run_episode(initial_state, thrust, gimbal, max_steps=500):
    state = initial_state
    history = []

    for _ in range(max_steps):
        history.append((state.x, state.y, state.theta))
        result = check_termination(state)
        if result:
            print(f"Episode ended: {result}")
            break
        state = step(state, thrust, gimbal)
    
    return history

# Matplotlib plotting for comparison plots/convergence curves
def plot_trajectory(history):
    xs, ys, thetas = zip(*history)
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys)
    plt.scatter(xs[0], ys[0], color='green', label='start')
    plt.scatter(xs[-1], ys[-1], color='red', label='end')
    plt.axhline(y=0, color='black', linewidth=2, label='ground')
    plt.axvline(x=0, color='gray', linestyle='--', label='target')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.title("Rocket Trajectory")
    plt.grid(True)
    plt.show()

# Pygame visualization for real-time simulation
def world_to_screen(x, y):
    screen_x = int(SCREEN_W / 2 + x * SCALE)
    screen_y = int(SCREEN_H - y * SCALE) # flip y axis
    return screen_x, screen_y

# Draw a simple rectangle to represent the rocket, centered on its position and rotated by its angle
def draw_rocket(screen, state):
    sx, sy = world_to_screen(state.x, state.y)
    pygame.draw.rect(screen, (200, 200, 200), (sx - 5, sy - 15, 10, 30))

# Run the simulation with Pygame visualization, stepping through the policy controls
def run_visual(initial_state, policy, max_steps=500):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()

    state = initial_state
    
    for (thrust, gimbal) in policy:
        # Handle window close button
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
    
        # Step your existing simulation
        state = step(state, thrust, gimbal)
        if check_termination(state):
            break

        # Draw
        screen.fill((20, 20, 40))
        draw_rocket(screen, state)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# Example usage: a simple policy that applies constant thrust with no gimbal for 500 steps
rocket_state = RocketState(0, 300, 5, -20, 0.1, 0)
policy = [(100.0, 0.0)] * 500
run_visual(rocket_state, policy)
history = run_episode(rocket_state, 100, 0)
plot_trajectory(history)