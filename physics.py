"""2D rigid-body rocket physics with quadratic atmospheric drag and O-U wind."""

import numpy as np
from dataclasses import dataclass
from config import (
    ENGINE_IGNITION_THRESHOLD,
    ENGINE_OFF_THRESHOLD,
    MIN_THROTTLE,
    ROCKET_HITBOX_HEIGHT,
    ROCKET_HITBOX_WIDTH,
)


@dataclass
class RocketState:
    """Full state of the rocket at a single time-step."""
    x: float = 0.0
    y: float = 200.0
    vx: float = 0.0
    vy: float = 0.0
    theta: float = 0.0   # angle from vertical (rad); 0 = upright, + = tilted right
    omega: float = 0.0   # angular velocity (rad/s)
    fuel: float = 1.0    # normalized remaining fuel [0, 1]

    def copy(self):
        return RocketState(self.x, self.y, self.vx, self.vy,
                           self.theta, self.omega, self.fuel)


class WindModel:
    """Ornstein–Uhlenbeck process for slowly varying horizontal wind."""

    def __init__(self, ou_theta: float = 0.3, max_wind: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.ou_theta = ou_theta
        self.max_wind = max_wind
        self.rng = rng or np.random.default_rng()
        # σ chosen so 3σ of the stationary distribution ≈ max_wind
        self.ou_sigma = (max_wind * np.sqrt(2.0 * ou_theta) / 3.0
                         if max_wind > 0 else 0.0)
        self.speed = 0.0

    def reset(self):
        self.speed = 0.0

    def step(self, dt: float) -> float:
        if self.ou_sigma > 0:
            self.speed += (self.ou_theta * (-self.speed) * dt
                           + self.ou_sigma * np.sqrt(dt) * self.rng.standard_normal())
        return self.speed


# ── body vertices in local coords (meters) for full OBB collision ──
# Rectangular hitbox only. Origin at base center, +y = up.
_BODY_VERTS = np.array([
    [-ROCKET_HITBOX_WIDTH / 2, ROCKET_HITBOX_HEIGHT],  # top-left
    [-ROCKET_HITBOX_WIDTH / 2, 0],                     # bottom-left
    [ROCKET_HITBOX_WIDTH / 2, 0],                      # bottom-right
    [ROCKET_HITBOX_WIDTH / 2, ROCKET_HITBOX_HEIGHT],   # top-right
])

# ── ground-contact parameters ──────────────────────────────────────
GROUND_FRICTION = 8.0       # horizontal deceleration on ground (m/s²)
GROUND_ANGULAR_DAMP = 2.0   # energy loss per radian/s on contact
SETTLE_TIME = 4.0           # max seconds on ground before forcing resolution
MIN_GROUND_TIME = 0.3       # minimum time before at-rest check fires
SETTLED_SPEED = 0.5         # speed + ω threshold for "at rest"
LANDED_SPEED = 8.0          # max speed at rest to count as landed
LANDED_TILT = np.radians(25)  # max tilt at rest to count as landed
EXPLODE_SPEED = 25.0        # impact speed above this → explosion
CG_HEIGHT = ROCKET_HITBOX_HEIGHT / 2  # CG at local (0, CG_HEIGHT)


def _vertex_world_offset(lx: float, ly: float, theta: float) -> tuple[float, float]:
    """Rotate a local vertex by theta; return world-frame offset from CG origin."""
    s, c = np.sin(theta), np.cos(theta)
    return lx * c + ly * s, -lx * s + ly * c


class RocketSim:
    """Full 2D rocket-landing simulation (one episode)."""

    def __init__(self, gravity: float, dry_mass: float, fuel_mass: float,
                 moi: float,
                 max_thrust: float, max_gimbal: float, thruster_arm: float,
                 fuel_rate: float, air_density: float, drag_cd_a: float,
                 dt: float, max_time: float, pad_x: float,
                 pad_half_width: float, wind_model: WindModel):
        self.gravity = gravity
        self.dry_mass = dry_mass
        self.fuel_mass = fuel_mass
        self.wet_mass = dry_mass + fuel_mass
        self.base_moi = moi
        self.max_thrust = max_thrust
        self.max_gimbal = max_gimbal
        self.thruster_arm = thruster_arm
        self.fuel_rate = fuel_rate
        self.air_density = air_density
        self.drag_cd_a = drag_cd_a
        self.dt = dt
        self.max_time = max_time
        self.max_steps = int(max_time / dt)
        self.pad_x = pad_x
        self.pad_half_width = pad_half_width
        self.wind = wind_model

        self.state = RocketState()
        self.time = 0.0
        self.steps = 0
        self.done = False
        self.frozen = False
        self.landed = False
        self.exploded = False
        self.on_ground = False
        self.ground_time = 0.0
        self.impact_speed = 0.0
        self.current_wind = 0.0
        self.engine_active = False
        self.engine_ever_on = False
        self.engine_shutdown = False

        # ── hinge state (Option A pivot constraint) ──
        # When on_ground, the body rotates about a fixed world point.
        # pivot_idx = local vertex index, pivot_world = (x, y) world coords.
        self.pivot_idx: int | None = None
        self.pivot_world: tuple[float, float] = (0.0, 0.0)

    def _mass_properties(self, fuel: float) -> tuple[float, float]:
        """Return current mass and moment of inertia from normalized fuel."""
        fuel = float(np.clip(fuel, 0.0, 1.0))
        mass = self.dry_mass + fuel * self.fuel_mass
        moi = self.base_moi * mass / self.wet_mass
        return mass, moi

    def _resolve_engine_throttle(self, throttle_cmd: float) -> float:
        """Apply single-use ignition, minimum throttle, and final cutoff."""
        throttle_cmd = float(np.clip(throttle_cmd, 0.0, 1.0))

        if self.engine_shutdown:
            self.engine_active = False
            return 0.0

        if not self.engine_ever_on:
            if throttle_cmd > ENGINE_IGNITION_THRESHOLD:
                self.engine_ever_on = True
                self.engine_active = True
            else:
                self.engine_active = False
                return 0.0
        else:
            if not self.engine_active or throttle_cmd < ENGINE_OFF_THRESHOLD:
                self.engine_active = False
                self.engine_shutdown = True
                return 0.0

        return max(MIN_THROTTLE, throttle_cmd)

    def reset(self, initial_state: RocketState) -> RocketState:
        self.state = initial_state.copy()
        self.time = 0.0
        self.steps = 0
        self.done = False
        self.frozen = False
        self.landed = False
        self.exploded = False
        self.on_ground = False
        self.ground_time = 0.0
        self.impact_speed = 0.0
        self.current_wind = 0.0
        self.engine_active = False
        self.engine_ever_on = False
        self.engine_shutdown = False
        self.pivot_idx = None
        self.pivot_world = (0.0, 0.0)
        self.wind.reset()
        return self.state

    def _lowest_vertex(self, x: float, y: float, theta: float):
        """Return (index, world_x, world_y) of the lowest body vertex."""
        s, c = np.sin(theta), np.cos(theta)
        # world y offset for each local vertex
        dys = -_BODY_VERTS[:, 0] * s + _BODY_VERTS[:, 1] * c
        idx = int(np.argmin(dys))
        lx, ly = _BODY_VERTS[idx]
        wx = x + lx * c + ly * s
        wy = y + dys[idx]
        return idx, float(wx), float(wy), float(dys[idx])

    def step(self, throttle: float, gimbal: float) -> tuple[bool, dict]:
        """Advance physics by one time-step."""
        if self.frozen:
            return True, {"wind": self.current_wind, "throttle": 0,
                          "gimbal_rad": 0, "thrust": 0,
                          "engine_active": self.engine_active,
                          "result": "exploded" if self.exploded else "frozen"}

        s = self.state
        throttle_cmd = float(np.clip(throttle, 0.0, 1.0))
        throttle = self._resolve_engine_throttle(throttle_cmd)
        gimbal_rad = float(np.clip(gimbal, -1.0, 1.0)) * self.max_gimbal
        mass, moi = self._mass_properties(s.fuel)

        # --- thrust (cut entirely once on the ground) ---
        if s.fuel <= 0 or self.on_ground:
            thrust_force = 0.0
            actual_throttle = 0.0
            self.engine_active = False
        else:
            thrust_force = throttle * self.max_thrust
            actual_throttle = throttle

        fire_angle = s.theta + gimbal_rad
        sin_fa, cos_fa = np.sin(fire_angle), np.cos(fire_angle)
        thrust_ax = (thrust_force / mass) * sin_fa
        thrust_ay = (thrust_force / mass) * cos_fa

        # --- quadratic drag: F = -0.5 ρ |v| v Cd_A ---
        speed_sq = s.vx * s.vx + s.vy * s.vy
        if speed_sq > 1e-12:
            speed = np.sqrt(speed_sq)
            drag_k = 0.5 * self.air_density * self.drag_cd_a * speed / mass
            drag_ax = -drag_k * s.vx
            drag_ay = -drag_k * s.vy
        else:
            drag_ax = drag_ay = 0.0

        # --- wind (horizontal force) ---
        w = self.wind.step(self.dt)
        self.current_wind = w
        wind_force = 0.5 * self.air_density * self.drag_cd_a * w * abs(w)
        wind_ax = wind_force / mass

        # gimbal torque → angular acceleration (thrust-driven, always applies)
        alpha = -(thrust_force * np.sin(gimbal_rad) * self.thruster_arm) / moi

        if self.on_ground and self.pivot_idx is not None:
            # ── HINGE DYNAMICS ─────────────────────────────────────
            # Body rotates about fixed world point self.pivot_world.
            # CG kinematics are derived from ω about the pivot; we do
            # NOT integrate linear velocity independently here.

            # Gravity torque about pivot: τ = r × F_g where r = CG - pivot
            # CG is at local (0, CG_HEIGHT) → world offset from rocket origin
            sin_t0, cos_t0 = np.sin(s.theta), np.cos(s.theta)
            cg_offset_x = CG_HEIGHT * sin_t0
            cg_offset_y = CG_HEIGHT * cos_t0
            cg_world_x = s.x + cg_offset_x
            cg_world_y = s.y + cg_offset_y
            rx = cg_world_x - self.pivot_world[0]
            ry = cg_world_y - self.pivot_world[1]
            # τ_gravity = rx * (-mg) - ry * 0 = -m·g·rx  (about pivot, z-axis)
            # With our sign convention (+θ = tilt right), τ_z → -α so:
            alpha += (mass * self.gravity * rx) / moi

            # angular damping
            alpha -= GROUND_ANGULAR_DAMP * s.omega

            # Integrate angular state
            new_omega = s.omega + alpha * self.dt
            new_theta = s.theta + new_omega * self.dt

            # Enforce hinge: place CG so pivot vertex stays at pivot_world
            lx_p, ly_p = _BODY_VERTS[self.pivot_idx]
            sin_t, cos_t = np.sin(new_theta), np.cos(new_theta)
            # world pos of pivot vertex = rocket_origin + R·(lx_p, ly_p)
            # → rocket_origin = pivot_world - R·(lx_p, ly_p)
            new_x = self.pivot_world[0] - (lx_p * cos_t + ly_p * sin_t)
            new_y = self.pivot_world[1] - (-lx_p * sin_t + ly_p * cos_t)

            # CG velocity from rigid-body rotation: v = ω × r
            new_cg_x = new_x + CG_HEIGHT * sin_t
            new_cg_y = new_y + CG_HEIGHT * cos_t
            rx_new = new_cg_x - self.pivot_world[0]
            ry_new = new_cg_y - self.pivot_world[1]
            new_vx = -new_omega * ry_new
            new_vy = new_omega * rx_new

            # Fuel still drains if throttle applied (even though thrust=0 on ground).
            # Since we cut thrust on ground, actual_throttle=0, so no drain.
            new_fuel = max(0.0, s.fuel - actual_throttle * self.fuel_rate * self.dt)

            # ── pivot switch / release check ──
            # After rotation, is a *different* vertex now lower than the current
            # pivot? If so, and it's at or below ground, switch pivots.
            idx_low, wx_low, wy_low, _ = self._lowest_vertex(new_x, new_y, new_theta)
            if idx_low != self.pivot_idx and wy_low <= 0.0:
                # Switch hinge to the new lowest vertex, snapped to ground.
                self.pivot_idx = idx_low
                self.pivot_world = (wx_low, 0.0)
                # Re-enforce hinge with new pivot
                lx_p, ly_p = _BODY_VERTS[self.pivot_idx]
                new_x = self.pivot_world[0] - (lx_p * cos_t + ly_p * sin_t)
                new_y = self.pivot_world[1] - (-lx_p * sin_t + ly_p * cos_t)
                new_cg_x = new_x + CG_HEIGHT * sin_t
                new_cg_y = new_y + CG_HEIGHT * cos_t
                rx_new = new_cg_x - self.pivot_world[0]
                ry_new = new_cg_y - self.pivot_world[1]
                new_vx = -new_omega * ry_new
                new_vy = new_omega * rx_new

        else:
            # ── FREE-FLIGHT DYNAMICS (unchanged semi-implicit Euler) ──
            ax = thrust_ax + drag_ax + wind_ax
            ay = thrust_ay + drag_ay - self.gravity

            new_vx = s.vx + ax * self.dt
            new_vy = s.vy + ay * self.dt
            new_x = s.x + new_vx * self.dt
            new_y = s.y + new_vy * self.dt
            new_omega = s.omega + alpha * self.dt
            new_theta = s.theta + new_omega * self.dt
            new_fuel = max(0.0, s.fuel - actual_throttle * self.fuel_rate * self.dt)

            # Ground collision detection
            idx_low, wx_low, wy_low, min_dy = self._lowest_vertex(
                new_x, new_y, new_theta)
            if wy_low <= 0.0:
                # First contact — establish hinge at the impact vertex.
                self.on_ground = True
                self.ground_time = 0.0
                self.impact_speed = np.sqrt(new_vx ** 2 + new_vy ** 2)
                self.pivot_idx = idx_low
                # Snap the rocket up so the contact vertex sits exactly on y=0,
                # and lock that world point as the hinge.
                self.pivot_world = (wx_low, 0.0)
                new_y = -min_dy  # lift CG so lowest vertex is at y=0
                # Kill vertical velocity; horizontal becomes tangential to pivot
                # (energy loss modelled via GROUND_ANGULAR_DAMP going forward)
                # Convert linear velocity into angular velocity contribution
                # about the new pivot for a cleaner impact response:
                # v_cg = ω × r_cg_from_pivot  →  ω = (r × v) / |r|²
                cg_wx = new_x + CG_HEIGHT * np.sin(new_theta)
                cg_wy = new_y + CG_HEIGHT * np.cos(new_theta)
                rx0 = cg_wx - self.pivot_world[0]
                ry0 = cg_wy - self.pivot_world[1]
                r2 = rx0 * rx0 + ry0 * ry0
                if r2 > 1e-9:
                    omega_from_impact = (rx0 * new_vy - ry0 * new_vx) / r2
                    # Blend: keep some of pre-impact ω, add impact contribution,
                    # but damp to reflect energy loss.
                    new_omega = 0.5 * (new_omega + omega_from_impact)
                new_vx = -new_omega * ry0
                new_vy = new_omega * rx0

        self.state = RocketState(new_x, new_y, new_vx, new_vy,
                                 new_theta, new_omega, new_fuel)
        self.time += self.dt
        self.steps += 1

        info = {
            "wind": w,
            "throttle": actual_throttle,
            "throttle_cmd": throttle_cmd,
            "gimbal_rad": gimbal_rad,
            "thrust": thrust_force,
            "mass": mass,
            "fuel_mass": new_fuel * self.fuel_mass,
            "engine_active": self.engine_active,
            "engine_ever_on": self.engine_ever_on,
            "engine_shutdown": self.engine_shutdown,
        }

        # --- outcome / termination ---
        if self.on_ground:
            self.ground_time += self.dt
            ground_speed = np.sqrt(new_vx ** 2 + new_vy ** 2)

            # High-impact explosion — freeze immediately
            if not self.done and self.impact_speed >= EXPLODE_SPEED:
                self.done = True
                self.frozen = True
                self.exploded = True
                self.landed = False
                info["result"] = "exploded"
                return True, info

            body_still = (ground_speed < SETTLED_SPEED
                          and abs(new_omega) < 0.1)
            at_rest = (self.ground_time >= MIN_GROUND_TIME and body_still)

            if not self.done and (at_rest or self.ground_time >= SETTLE_TIME):
                self.done = True
                on_pad = abs(new_x - self.pad_x) <= self.pad_half_width
                upright = abs(new_theta) < LANDED_TILT
                slow = ground_speed < LANDED_SPEED
                self.landed = on_pad and upright and slow
                info["result"] = "landed" if self.landed else "crashed"

            if self.done and not self.frozen:
                info["result"] = "landed" if self.landed else "crashed"
                if body_still or self.ground_time >= SETTLE_TIME + 2.0:
                    self.frozen = True

        elif self.steps >= self.max_steps:
            self.done = True
            self.frozen = True
            info["result"] = "timeout"
        elif abs(new_x) > 500 or new_y > 600:
            self.done = True
            self.frozen = True
            info["result"] = "out_of_bounds"

        return self.done, info