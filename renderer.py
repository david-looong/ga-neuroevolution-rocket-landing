"""Pygame visualisation for rocket-landing replay after each generation.

Shows all rockets simultaneously: the best genome in full colour, others at
~20% opacity.  The preview continues for ~1 s after the best rocket settles.
"""

import numpy as np

try:
    from pathlib import Path
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from controller import NeuralNetwork


# ═══════════════════════════════════════════════════════════════════
#  Exhaust particle system
# ═══════════════════════════════════════════════════════════════════

class ExhaustParticle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size")

    def __init__(self, x, y, vx, vy, life, size):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size


class ExhaustSystem:
    """Per-rocket particle emitter for thruster exhaust."""

    HOT_CORE   = np.array([255, 255, 220], dtype=float)
    MID_FLAME  = np.array([255, 180, 50],  dtype=float)
    COOL_TIP   = np.array([255, 60,  15],  dtype=float)
    SMOKE      = np.array([80,  80,  90],  dtype=float)

    def __init__(self):
        self.particles: list[ExhaustParticle] = []
        self._rng = np.random.default_rng()

    def emit(self, sx, sy, throttle, gimbal_rad, theta, scale):
        """Spawn new particles at the nozzle (screen coords)."""
        if throttle < 0.02:
            return

        rng = self._rng
        n = max(1, int(throttle * 8 + rng.uniform(0, 2)))

        fa = theta + gimbal_rad
        sin_f = np.sin(fa)
        cos_f = np.cos(fa)

        base_speed = (40 + throttle * 120) * scale / 60.0

        for _ in range(n):
            spread = rng.uniform(-0.25, 0.25)
            a = fa + spread
            sa, ca = np.sin(a), np.cos(a)

            spd = base_speed * rng.uniform(0.6, 1.3)
            vx = -sa * spd
            vy =  ca * spd

            ox = rng.uniform(-2.5, 2.5) * cos_f
            oy = rng.uniform(-2.5, 2.5) * sin_f

            life = rng.uniform(0.15, 0.4 + throttle * 0.2)
            size = rng.uniform(1.5, 3.0 + throttle * 2.0)
            self.particles.append(
                ExhaustParticle(sx + ox, sy + oy, vx, vy, life, size))

    def emit_explosion(self, sx, sy, impact_speed, scale):
        """Burst of debris particles on high-impact crash."""
        rng = self._rng
        intensity = min(impact_speed / 20.0, 3.0)
        n = int(40 + intensity * 30)

        for _ in range(n):
            angle = rng.uniform(0, 2 * np.pi)
            spd = rng.uniform(1.0, 4.0 + intensity * 0.5) * scale
            vx = np.cos(angle) * spd
            vy = np.sin(angle) * spd - rng.uniform(0.5, 2.0) * scale

            life = rng.uniform(0.4, 1.2 + intensity * 0.3)
            size = rng.uniform(2.0, 4.0 + intensity * 1.0)
            ox = rng.uniform(-6, 6)
            oy = rng.uniform(-6, 6)
            self.particles.append(
                ExhaustParticle(sx + ox, sy + oy, vx, vy, life, size))

    def update(self, dt):
        frame_scale = dt * 60.0
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.x += p.vx * frame_scale
            p.y += p.vy * frame_scale
            p.vx *= 0.96 ** frame_scale
            p.vy *= 0.96 ** frame_scale
            p.size *= 0.985 ** frame_scale
            alive.append(p)
        self.particles = alive

    def draw(self, surface, alpha=255):
        for p in self.particles:
            t = 1.0 - (p.life / p.max_life)
            if t < 0.3:
                clr = self.HOT_CORE + (self.MID_FLAME - self.HOT_CORE) * (t / 0.3)
            elif t < 0.7:
                clr = self.MID_FLAME + (self.COOL_TIP - self.MID_FLAME) * ((t - 0.3) / 0.4)
            else:
                clr = self.COOL_TIP + (self.SMOKE - self.COOL_TIP) * ((t - 0.7) / 0.3)

            a = int(alpha * (1.0 - t * 0.8))
            c = (int(clr[0]), int(clr[1]), int(clr[2]), max(0, min(255, a)))
            sz = max(1, int(p.size * (1.0 - t * 0.5)))
            rect = pygame.Rect(int(p.x) - sz, int(p.y) - sz, sz * 2, sz * 2)
            pygame.draw.rect(surface, c, rect)


# ═══════════════════════════════════════════════════════════════════
#  Renderer
# ═══════════════════════════════════════════════════════════════════

class Renderer:
    """Opens a persistent window and replays episodes between generations."""

    BACKGROUND_PIXEL_GROUND_HEIGHT = 8
    HUD_PANEL_MARGIN = 16
    HUD_PANEL_WIDTH = 260
    HUD_PAD = 18
    DEFAULT_CAMERA_ZOOM = 4.0
    MAX_CAMERA_ZOOM = 8.0
    ZOOM_STEP = 0.25
    REFERENCE_PIXEL_SCALE = 4
    UI_PIXEL_SCALE = 4

    # ── colour palette ──────────────────────────────────────────────
    BG        = (8, 8, 32)
    TRAIL     = (80, 140, 255)
    TRAIL_DIM = (40, 70, 130)
    TRAIL_HOT = (165, 220, 255)
    HUD       = (190, 255, 190)
    HUD_DIM   = (120, 120, 120)
    METER_FUEL = (80, 220, 120)
    METER_FUEL_DARK = (32, 120, 70)
    METER_THRUST = (255, 170, 64)
    METER_THRUST_DARK = (150, 72, 32)

    GHOST_ALPHA = 80
    TRAIL_MAX_SEGMENTS = 180
    GHOST_TRAIL_MAX_SEGMENTS = 90
    MAX_PHYSICS_STEPS_PER_FRAME = 8

    def __init__(self, width: int = 1440, height: int = 800,
                 scale: float = 2.4, fps: int = 60):
        if not HAS_PYGAME:
            raise RuntimeError("pygame is required for visualisation")
        pygame.init()
        pygame.display.set_caption("2D Rocket Landing — Genetic Algorithm")
        self.W = width
        self.H = height
        self.base_vis_scale = scale
        self.scale = scale
        self.fps = fps
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.sprite_rng = np.random.default_rng()
        self.dragging_camera = False
        self.last_mouse_pos = (0, 0)
        self._load_background()
        self._init_camera()
        self._init_fonts()
        self._load_ui_sprite()
        self._load_button_sprites()
        self._load_rocket_sprites()
        self._init_particles()

    # ── public API ──────────────────────────────────────────────────

    def replay_generation(self, best_idx, all_genomes, nn_layers, make_sim,
                          gen_info, normalize_fn, _diverse_genomes=None):
        """Replay all genomes simultaneously from one generation.

        Parameters
        ----------
        best_idx      : int — population index of the best genome
        all_genomes   : list[(pop_idx, genome)] — genomes to show
        nn_layers     : list[int]
        make_sim      : callable() → (RocketSim, RocketState)
        gen_info      : dict with keys gen, total_gen, fitness, success_rate
        normalize_fn  : callable(RocketState) → np.ndarray(6,)
        _diverse_genomes : list[(label, genome)] or None — reserved for UI use

        Returns
        -------
        "quit" | "skip" | "done"
        """
        n = len(all_genomes)

        nns = [NeuralNetwork(nn_layers) for _ in range(n)]
        sims = []
        trails = [[] for _ in range(n)]
        exhausts = [ExhaustSystem() for _ in range(n)]
        exploded = [False] * n
        rocket_sprite_idxs = self.sprite_rng.integers(
            0, len(self.rocket_sprites), size=n)
        last_infos = [{"wind": 0, "throttle": 0, "gimbal_rad": 0, "thrust": 0}
                      for _ in range(n)]
        is_best = [False] * n

        for j, (pop_idx, genome) in enumerate(all_genomes):
            nns[j].set_genome(genome)
            sim, initial = make_sim()
            sim.reset(initial)
            sims.append(sim)
            if pop_idx == best_idx:
                is_best[j] = True

        best_j = next((j for j in range(n) if is_best[j]), 0)
        is_best = [j == best_j for j in range(n)]

        # ── simulation phase: step all rockets each frame ──────────
        best_frozen_ms = None
        POST_REST_MS = 1000
        frame_dt = 1.0 / max(self.fps, 1)
        sim_accum = 0.0
        sim_step_dt = sims[0].dt if sims else frame_dt

        while True:
            frame_ms = min(self.clock.tick(self.fps), 100)
            frame_dt = frame_ms / 1000.0 if frame_ms > 0 else frame_dt
            sim_accum += frame_dt
            action = self._poll()
            if action == "quit":
                return "quit"
            if action == "skip":
                return "skip"

            physics_steps = 0
            while (sim_accum >= sim_step_dt
                   and physics_steps < self.MAX_PHYSICS_STEPS_PER_FRAME):
                sim_accum -= sim_step_dt
                physics_steps += 1

                for j in range(n):
                    if sims[j].frozen:
                        continue

                    # NN only drives the rocket until outcome is determined
                    if not sims[j].done:
                        inp = normalize_fn(sims[j].state)
                        throttle, gimbal = nns[j].forward(inp)
                    else:
                        throttle, gimbal = 0.0, 0.0

                    _, info = sims[j].step(throttle, gimbal)
                    last_infos[j] = info
                    trails[j].append((sims[j].state.x, sims[j].state.y))

                    # Explosion trigger
                    if sims[j].exploded and not exploded[j]:
                        exploded[j] = True
                        sx, sy = self._w2s(sims[j].state.x, sims[j].state.y)
                        exhausts[j].emit_explosion(
                            sx, sy, sims[j].impact_speed, self.scale)

                    # Exhaust particles for active thrust
                    if not sims[j].exploded:
                        sx, sy = self._w2s(sims[j].state.x, sims[j].state.y)
                        exhausts[j].emit(sx, sy,
                                         info.get("throttle", 0.0),
                                         info.get("gimbal_rad", 0.0),
                                         sims[j].state.theta, self.scale)

            if physics_steps >= self.MAX_PHYSICS_STEPS_PER_FRAME:
                sim_accum = min(sim_accum, sim_step_dt)

            for exhaust in exhausts:
                exhaust.update(frame_dt)

            if sims[best_j].frozen and best_frozen_ms is None:
                best_frozen_ms = 0

            if best_frozen_ms is not None:
                best_frozen_ms += frame_ms

            all_frozen = all(sim.frozen for sim in sims)
            best_rested = best_frozen_ms is not None and best_frozen_ms >= POST_REST_MS

            if all_frozen or best_rested:
                break

            self._draw_all(sims, trails, exhausts, last_infos,
                           is_best, best_j, gen_info, rocket_sprite_idxs,
                           frame_dt=frame_dt, exploded_flags=exploded)

        # ── end-of-episode: show result overlay ────────────────────
        bs = sims[best_j]
        if bs.landed:
            result_txt, result_clr = "LANDED!", (80, 255, 100)
        elif bs.exploded:
            result_txt, result_clr = "EXPLODED", (255, 100, 30)
        elif last_infos[best_j].get("result") == "timeout":
            result_txt, result_clr = "TIMEOUT", (255, 200, 60)
        else:
            result_txt, result_clr = "CRASHED", (255, 80, 80)
        result_breakdown = self._fitness_breakdown(bs)

        wait = 0
        hold_result = False
        frame_dt = 1.0 / max(self.fps, 1)
        while hold_result or wait < 1500:
            frame_ms = min(self.clock.tick(self.fps), 100)
            frame_dt = frame_ms / 1000.0 if frame_ms > 0 else frame_dt
            action = self._poll()
            if action == "quit":
                return "quit"
            if action in ("skip", "next"):
                break
            if action == "hold_result":
                hold_result = not hold_result
                continue
            # Keep updating particles during overlay
            for j in range(n):
                exhausts[j].update(frame_dt)
            self._draw_all(sims, trails, exhausts, last_infos,
                           is_best, best_j, gen_info,
                           rocket_sprite_idxs,
                           overlay=result_txt, overlay_clr=result_clr,
                           overlay_breakdown=result_breakdown,
                           overlay_hold=hold_result,
                           frame_dt=frame_dt, exploded_flags=exploded)
            if not hold_result:
                wait += frame_ms

        return "done"

    def close(self):
        pygame.quit()

    # ── event handling ──────────────────────────────────────────────

    def _poll(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "quit"
            if ev.type == pygame.MOUSEWHEEL:
                self._set_camera_zoom(
                    self.camera_zoom + self.ZOOM_STEP * ev.y,
                    pygame.mouse.get_pos())
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button in (1, 2, 3):
                    self.dragging_camera = True
                    self.last_mouse_pos = ev.pos
            if ev.type == pygame.MOUSEBUTTONUP:
                if ev.button in (1, 2, 3):
                    self.dragging_camera = False
            if ev.type == pygame.MOUSEMOTION and self.dragging_camera:
                dx = ev.pos[0] - self.last_mouse_pos[0]
                dy = ev.pos[1] - self.last_mouse_pos[1]
                self.camera_x -= dx / self.pixel_art_scale
                self.camera_y -= dy / self.pixel_art_scale
                self.last_mouse_pos = ev.pos
                self._clamp_camera()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    return "skip"
                if ev.key == pygame.K_n:
                    return "next"
                if ev.key == pygame.K_h:
                    return "hold_result"
                if ev.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    self._set_camera_zoom(self.camera_zoom + self.ZOOM_STEP)
                if ev.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    self._set_camera_zoom(self.camera_zoom - self.ZOOM_STEP)
                if ev.key == pygame.K_0:
                    self._reset_camera()
        return None

    # ── coordinate helpers ──────────────────────────────────────────

    def _w2s(self, wx, wy):
        """World → screen."""
        bx = self.bg_native_size[0] / 2 + wx * self.world_source_px_per_meter
        by = self.ground_y_src - wy * self.world_source_px_per_meter
        sx = self.W / 2 + (bx - self.camera_x) * self.pixel_art_scale
        sy = self.H / 2 + (by - self.camera_y) * self.pixel_art_scale
        return int(sx), int(sy)

    # ── particle pool for wind visualisation ────────────────────────

    def _init_particles(self):
        rng = np.random.default_rng(0)
        self.wind_parts = [[rng.uniform(0, self.bg_native_size[0]),
                            rng.uniform(0, self.ground_y_src),
                            rng.uniform(0.3, 1.0)]
                           for _ in range(80)]

    def _load_rocket_sprites(self):
        assets_dir = Path(__file__).resolve().parent / "assets"
        sprite_paths = sorted(assets_dir.glob("rocket*.png"))
        if not sprite_paths:
            raise FileNotFoundError(f"No rocket sprites found in {assets_dir}")

        self.rocket_source_sprites = [
            pygame.image.load(str(sprite_path)).convert_alpha()
            for sprite_path in sprite_paths
        ]
        self._rescale_rocket_sprites()

    def _load_ui_sprite(self):
        ui_path = Path(__file__).resolve().parent / "assets" / "ui.png"
        self.ui_sprite = pygame.image.load(str(ui_path)).convert_alpha()
        self.ui_corner_px = 4
        self.ui_corner_screen_px = max(1, int(self.ui_corner_px * self.ui_scale))

    def _load_button_sprites(self):
        assets_dir = Path(__file__).resolve().parent / "assets"
        self.button_sprite = self._load_optional_ui_sprite(assets_dir / "button.png")
        self.button_pressed_sprite = self._load_optional_ui_sprite(
            assets_dir / "button_pressed.png")

    def _load_optional_ui_sprite(self, path):
        return pygame.image.load(str(path)).convert_alpha() if path.exists() else None

    def _load_background(self):
        assets_dir = Path(__file__).resolve().parent / "assets"
        bg_path = assets_dir / "background_2.png"
        if not bg_path.exists():
            bg_path = assets_dir / "background.png"
        self.background_source = pygame.image.load(str(bg_path)).convert()
        self.bg_native_size = self.background_source.get_size()
        self.ground_y_src = self.bg_native_size[1] - self.BACKGROUND_PIXEL_GROUND_HEIGHT

    def _init_camera(self):
        self.ui_scale = self.UI_PIXEL_SCALE
        self.world_source_px_per_meter = (
            self.base_vis_scale / self.REFERENCE_PIXEL_SCALE)
        self.min_camera_zoom = max(
            self.W / self.bg_native_size[0],
            self.H / self.bg_native_size[1],
        )
        self.camera_zoom = max(self.DEFAULT_CAMERA_ZOOM, self.min_camera_zoom)
        self.pixel_art_scale = self.camera_zoom
        self.camera_x = self.bg_native_size[0] / 2
        self.camera_y = self.bg_native_size[1] / 2
        self._apply_camera_zoom()

    def _apply_camera_zoom(self):
        self.pixel_art_scale = self.camera_zoom
        self.scale = self.world_source_px_per_meter * self.pixel_art_scale
        bg_w = max(1, int(round(self.bg_native_size[0] * self.pixel_art_scale)))
        bg_h = max(1, int(round(self.bg_native_size[1] * self.pixel_art_scale)))
        self.background = pygame.transform.scale(self.background_source, (bg_w, bg_h))
        self.background_size = (bg_w, bg_h)
        self._clamp_camera()
        if hasattr(self, "rocket_source_sprites"):
            self._rescale_rocket_sprites()

    def _reset_camera(self):
        self.camera_x = self.bg_native_size[0] / 2
        self.camera_y = self.bg_native_size[1] / 2
        self._set_camera_zoom(max(self.DEFAULT_CAMERA_ZOOM, self.min_camera_zoom))

    def _set_camera_zoom(self, zoom, focus_screen=None):
        zoom = float(np.clip(zoom, self.min_camera_zoom, self.MAX_CAMERA_ZOOM))
        if abs(zoom - self.camera_zoom) < 1e-6:
            return

        old_zoom = self.pixel_art_scale
        if focus_screen is None:
            focus_screen = (self.W / 2, self.H / 2)
        fx, fy = focus_screen
        focus_src_x = self.camera_x + (fx - self.W / 2) / old_zoom
        focus_src_y = self.camera_y + (fy - self.H / 2) / old_zoom

        self.camera_zoom = zoom
        self.pixel_art_scale = zoom
        self.camera_x = focus_src_x - (fx - self.W / 2) / self.pixel_art_scale
        self.camera_y = focus_src_y - (fy - self.H / 2) / self.pixel_art_scale
        self._apply_camera_zoom()

    def _clamp_camera(self):
        view_w = self.W / self.pixel_art_scale
        view_h = self.H / self.pixel_art_scale
        bg_w, bg_h = self.bg_native_size

        if view_w >= bg_w:
            self.camera_x = bg_w / 2
        else:
            half_w = view_w / 2
            self.camera_x = float(np.clip(self.camera_x, half_w, bg_w - half_w))

        if view_h >= bg_h:
            self.camera_y = bg_h / 2
        else:
            half_h = view_h / 2
            self.camera_y = float(np.clip(self.camera_y, half_h, bg_h - half_h))

    def _pan_camera(self, dt):
        keys = pygame.key.get_pressed()
        dx = ((keys[pygame.K_d] or keys[pygame.K_RIGHT])
              - (keys[pygame.K_a] or keys[pygame.K_LEFT]))
        dy = ((keys[pygame.K_s] or keys[pygame.K_DOWN])
              - (keys[pygame.K_w] or keys[pygame.K_UP]))
        if dx == 0 and dy == 0:
            return

        speed = 360.0 / max(self.pixel_art_scale, 1)
        self.camera_x += dx * speed * dt
        self.camera_y += dy * speed * dt
        self._clamp_camera()

    def _background_screen_pos(self):
        return (int(round(self.W / 2 - self.camera_x * self.pixel_art_scale)),
                int(round(self.H / 2 - self.camera_y * self.pixel_art_scale)))

    def _source_to_screen(self, bx, by):
        sx = self.W / 2 + (bx - self.camera_x) * self.pixel_art_scale
        sy = self.H / 2 + (by - self.camera_y) * self.pixel_art_scale
        return int(sx), int(sy)

    def _draw_trail(self, surface, trail, base_color, alpha, max_segments,
                    max_width, node_every=10, hot_color=None):
        if len(trail) < 2:
            return

        start = max(1, len(trail) - max_segments)
        visible_count = max(1, len(trail) - start)
        snap = max(1, int(round(self.pixel_art_scale)))
        hot_color = hot_color or base_color

        for i in range(start, len(trail)):
            age = (i - start + 1) / visible_count
            p1 = self._w2s(*trail[i - 1])
            p2 = self._w2s(*trail[i])
            if p1 == p2:
                continue

            width = max(1, int(round(max_width * (0.35 + age * 0.65))))
            segment_alpha = int(alpha * (age ** 1.6))
            color = tuple(
                int(base_color[k] + (hot_color[k] - base_color[k]) * (age ** 2))
                for k in range(3)
            ) + (segment_alpha,)
            pygame.draw.line(surface, color, p1, p2, width)

            if i % node_every == 0 or i == len(trail) - 1:
                node_size = max(1, int(round(snap * (0.35 + age * 0.65))))
                node_alpha = int(segment_alpha * 0.8)
                rect = pygame.Rect(p2[0] - node_size // 2, p2[1] - node_size // 2,
                                   node_size, node_size)
                pygame.draw.rect(surface, color[:3] + (node_alpha,), rect)

    def _rescale_rocket_sprites(self):
        self.rocket_sprites = []
        for sprite in self.rocket_source_sprites:
            sprite_w_px = max(1, int(round(sprite.get_width() * self.pixel_art_scale)))
            sprite_h_px = max(1, int(round(sprite.get_height() * self.pixel_art_scale)))
            scaled = pygame.transform.scale(sprite, (sprite_w_px, sprite_h_px))
            self.rocket_sprites.append((scaled, sprite_h_px))

    def _init_fonts(self):
        font_path = Path(__file__).resolve().parent / "assets" / "pixel.ttf"
        font_file = str(font_path) if font_path.exists() else None
        self.font_scale = 0.5
        self.font = pygame.font.Font(font_file, 22)
        self.font_lg = pygame.font.Font(font_file, 30)
        self.font_xl = pygame.font.Font(font_file, 48)

    def _render_pixel_text(self, font, text, color):
        text_surf = font.render(text, False, color)
        if self.font_scale == 1:
            return text_surf
        size = (max(1, int(round(text_surf.get_width() * self.font_scale))),
                max(1, int(round(text_surf.get_height() * self.font_scale))))
        return pygame.transform.scale(text_surf, size)

    def _trim_surface_alpha(self, surface):
        """Crop transparent font metric padding around rendered text."""
        bounds = surface.get_bounding_rect()
        if bounds.width == 0 or bounds.height == 0:
            return surface
        return surface.subsurface(bounds).copy()

    def _fitness_breakdown(self, sim):
        """Return replay fitness terms matching main.compute_fitness."""
        s = sim.state
        dx = abs(s.x - sim.pad_x)

        proximity = max(0.0, 1.0 - (dx / 200.0) ** 0.5) * 40.0
        descent = max(0.0, 1.0 - s.y / 300.0) * 20.0

        ground_gate = 1.0 if sim.on_ground else max(0.0, 1.0 - s.y / 50.0)
        tilt_deg = abs(np.degrees(s.theta))
        tilt_pts = max(0.0, 1.0 - abs(s.theta) / (np.pi / 2)) * ground_gate * 15.0

        #impact_pts = (max(0.0, 1.0 - sim.impact_speed / 30.0) * 40.0
        #              if sim.on_ground else 0.0)

        if sim.on_ground:
            v = sim.impact_speed
            # Base reward: exponential decay
            impact_pts = max(0.0, 1.0 - sim.impact_speed / 30.0) * 40.0
            # Tier bonuses for hitting milestones
            #if v < 10.0: impact_pts += 5.0   # "survivable"
            #if v < 5.0:  impact_pts += 10.0  # "soft landing"
            #if v < 2.0:  impact_pts += 15.0  # "feather landing"
            #if v < 0.5:  impact_pts += 20.0  # "perfect"
        else:
            impact_pts = 0.0

        fuel_used = 1.0 - s.fuel
        fuel_bonus = s.fuel * 40.0 if sim.landed else 0.0

        terms = [
            ("PROX", f"dx {dx:5.1f}m", proximity, 40.0),
            ("DESC", f"alt {s.y:5.1f}m", descent, 20.0),
            ("TILT", f"{tilt_deg:5.1f}deg", tilt_pts, 15.0),
            ("IMPACT", f"{sim.impact_speed:5.1f}m/s", impact_pts, 40.0),
            ("FUEL", f"used {fuel_used:4.0%}", fuel_bonus, 40.0),
        ]
        total = sum(points for _, _, points, _ in terms)
        return total, terms

    def _draw_ui_panel(self, surface, rect):
        """Draw assets/ui.png as a 9-slice panel with 4px source corners."""
        self._draw_nine_slice_scaled(
            surface, self.ui_sprite, rect, self.ui_corner_px, self.ui_corner_screen_px)

    def _draw_result_overlay(self, surface, title, title_color, breakdown,
                             eval_fitness=None, held=False):
        title_surf = self._trim_surface_alpha(
            self._render_pixel_text(self.font_xl, title, title_color))

        total, terms = breakdown
        total_surf = self._trim_surface_alpha(
            self._render_pixel_text(self.font_lg, f"REPLAY {total:6.1f}", self.HUD))
        eval_surf = None
        if eval_fitness is not None:
            eval_surf = self._trim_surface_alpha(
                self._render_pixel_text(
                    self.font, f"EVAL BEST {eval_fitness:6.1f}", self.HUD_DIM))
        hint = "HELD - H RELEASE  SPACE SKIP" if held else "H HOLD  SPACE SKIP"
        hint_surf = self._trim_surface_alpha(
            self._render_pixel_text(self.font, hint, self.HUD_DIM))
        term_surfs = []
        for name, value, points, max_points in terms:
            term_surfs.append((
                self._trim_surface_alpha(self._render_pixel_text(self.font, name, self.HUD)),
                self._trim_surface_alpha(self._render_pixel_text(self.font, value, self.HUD_DIM)),
                self._trim_surface_alpha(self._render_pixel_text(self.font, f"+{points:5.1f}", self.HUD)),
                points,
                max_points,
            ))

        gap = 8
        row_gap = 10
        pad_x = 22
        pad_y = 16
        bar_w = 92
        bar_h = 10
        label_w = max(label.get_width() for label, _, _, _, _ in term_surfs)
        detail_w = max(detail.get_width() for _, detail, _, _, _ in term_surfs)
        score_w = max(score.get_width() for _, _, score, _, _ in term_surfs)
        row_h = max(
            max(label.get_height(), detail.get_height(), score.get_height())
            for label, detail, score, _, _ in term_surfs
        )
        content_w = max(
            title_surf.get_width(),
            total_surf.get_width(),
            eval_surf.get_width() if eval_surf else 0,
            hint_surf.get_width(),
            label_w + 10 + detail_w + 12 + bar_w + 12 + score_w,
        )
        content_h = (title_surf.get_height() + gap
                     + total_surf.get_height() + gap)
        if eval_surf:
            content_h += eval_surf.get_height() + gap
        content_h += (
                     + len(term_surfs) * row_h
                     + (len(term_surfs) - 1) * row_gap
                     + gap + hint_surf.get_height())

        panel = pygame.Rect(0, 0, content_w + pad_x * 2, content_h + pad_y * 2)
        panel.center = (self.W // 2, 150)
        self._draw_ui_panel(surface, panel)

        y = panel.top + pad_y
        surface.blit(title_surf, (panel.centerx - title_surf.get_width() // 2, y))
        y += title_surf.get_height() + gap
        surface.blit(total_surf, (panel.centerx - total_surf.get_width() // 2, y))
        y += total_surf.get_height() + gap
        if eval_surf:
            surface.blit(eval_surf, (panel.centerx - eval_surf.get_width() // 2, y))
            y += eval_surf.get_height() + gap

        x = panel.left + pad_x
        detail_x = x + label_w + 10
        bar_x = detail_x + detail_w + 12
        score_x = bar_x + bar_w + 12
        for label, detail, score, points, max_points in term_surfs:
            surface.blit(label, (x, y))
            surface.blit(detail, (detail_x, y))

            bar_y = y + (row_h - bar_h) // 2
            fill_w = int(bar_w * np.clip(points / max_points, 0.0, 1.0))
            pygame.draw.rect(surface, (28, 35, 48), (bar_x, bar_y, bar_w, bar_h))
            if fill_w > 0:
                pygame.draw.rect(surface, title_color, (bar_x, bar_y, fill_w, bar_h))
                pygame.draw.rect(surface, self.HUD, (bar_x, bar_y, min(fill_w, 3), bar_h))
            surface.blit(score, (score_x, y))
            y += row_h + row_gap

        y += gap - row_gap
        surface.blit(hint_surf, (panel.centerx - hint_surf.get_width() // 2, y))

    # ── main composite draw routine ────────────────────────────────

    def _draw_all(self, sims, trails, exhausts, infos, is_best, best_j, gi,
                  rocket_sprite_idxs, overlay=None, overlay_clr=None,
                  overlay_breakdown=None, overlay_hold=False,
                  frame_dt=None, exploded_flags=None):
        frame_dt = frame_dt or (1.0 / max(self.fps, 1))
        self._pan_camera(frame_dt)

        scr = self.screen
        scr.fill(self.BG)
        scr.blit(self.background, self._background_screen_pos())
        ex_flags = exploded_flags or [False] * len(sims)

        # wind particles (use best sim's wind)
        wind = infos[best_j].get("wind", 0.0)
        frame_scale = frame_dt * 60.0
        for p in self.wind_parts:
            p[0] += wind * self.world_source_px_per_meter * frame_dt * 3
            p[1] += (np.random.uniform(-0.3, 0.3) * frame_scale
                     / max(self.pixel_art_scale, 1))
            if p[0] > self.bg_native_size[0]:
                p[0] = 0.0
            elif p[0] < 0:
                p[0] = self.bg_native_size[0]
            if p[1] > self.ground_y_src:
                p[1] = 0.0
            elif p[1] < 0:
                p[1] = self.ground_y_src
            bright = min(200, int(40 + abs(wind) * 12))
            sx, sy = self._source_to_screen(p[0], p[1])
            pygame.draw.circle(scr, (bright, bright + 5, bright + 15),
                               (sx, sy),
                               1 if abs(wind) < 5 else 2)

        # landing pad

        # ── draw ghost rockets first (behind best) ─────────────────
        ghost_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        for j in range(len(sims)):
            if is_best[j]:
                continue
            self._draw_trail(
                ghost_surf, trails[j], self.TRAIL_DIM, self.GHOST_ALPHA,
                self.GHOST_TRAIL_MAX_SEGMENTS, max_width=2, node_every=14)

            exhausts[j].draw(ghost_surf, alpha=self.GHOST_ALPHA)
            if not ex_flags[j]:
                self._draw_rocket_body(ghost_surf, sims[j].state,
                                       rocket_sprite_idxs[j],
                                       alpha=self.GHOST_ALPHA)

        scr.blit(ghost_surf, (0, 0))

        # ── draw best rocket exhaust behind body ───────────────────
        best_exhaust_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        exhausts[best_j].draw(best_exhaust_surf, alpha=255)
        scr.blit(best_exhaust_surf, (0, 0))

        # ── draw best rocket trail + body on top ───────────────────
        best_trail_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self._draw_trail(
            best_trail_surf, trails[best_j], self.TRAIL, 220,
            self.TRAIL_MAX_SEGMENTS, max_width=4, node_every=8,
            hot_color=self.TRAIL_HOT)
        scr.blit(best_trail_surf, (0, 0))

        if not ex_flags[best_j]:
            self._draw_rocket_body(scr, sims[best_j].state,
                                   rocket_sprite_idxs[best_j], alpha=255)

        # HUD
        self._draw_hud(sims[best_j].state, infos[best_j], gi)

        # overlay text
        if overlay:
            if overlay_breakdown is None:
                overlay_breakdown = self._fitness_breakdown(sims[best_j])
            self._draw_result_overlay(
                scr, overlay, overlay_clr, overlay_breakdown,
                eval_fitness=gi.get("fitness"), held=overlay_hold)

        pygame.display.flip()

    # ── rocket body (no flame — exhaust is handled by particle system) ──

    def _draw_rocket_body(self, surface, state, sprite_idx, alpha=255):
        """Draw the rocket sprite around a base-center pivot."""
        sx, sy = self._w2s(state.x, state.y)
        th = state.theta
        cos_t, sin_t = np.cos(th), np.sin(th)
        sprite, sprite_h_px = self.rocket_sprites[int(sprite_idx)]
        half_h_px = sprite_h_px / 2.0

        cx = sx + half_h_px * sin_t
        cy = sy - half_h_px * cos_t

        # Use rotate (not rotozoom) to avoid smoothing / blur on pixel art.
        rotated = pygame.transform.rotate(sprite, -np.degrees(th))

        if alpha < 255:
            # Apply per-surface alpha modulation after rotation so ghost rockets
            # remain transparent even on per-pixel-alpha source images.
            rotated = rotated.copy()
            rotated.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)

        rect = rotated.get_rect(center=(int(cx), int(cy)))
        surface.blit(rotated, rect)

    # ── heads-up display ────────────────────────────────────────────

    def _draw_meter(self, rect, value, color, dark_color):
        self._draw_ui_panel(self.screen, rect)
        inner_pad = max(4, int(self.ui_scale * 2))
        inner = pygame.Rect(rect).inflate(-inner_pad * 2, -inner_pad * 2)

        value = float(np.clip(value, 0.0, 1.0))
        fill_area = inner.inflate(-8, -8)
        filled_w = int(fill_area.width * value)
        snap = max(1, int(self.ui_scale))
        filled_w = (filled_w // snap) * snap
        filled = pygame.Rect(fill_area.left, fill_area.top, filled_w, fill_area.height)

        stripe_w = snap
        x = filled.left
        stripe_i = 0
        while x < filled.right:
            w = min(stripe_w, filled.right - x)
            clr = color if stripe_i % 2 == 0 else dark_color
            pygame.draw.rect(self.screen, clr, (x, filled.top, w, filled.height))
            x += stripe_w
            stripe_i += 1

    def _draw_button(self, rect, text):
        keys = pygame.key.get_pressed()
        pressed = keys[pygame.K_SPACE]
        sprite = (self.button_pressed_sprite if pressed and self.button_pressed_sprite
                  else self.button_sprite)
        if sprite is None:
            self._draw_ui_panel(self.screen, rect)
        else:
            self._draw_nine_slice(self.screen, sprite, rect, 4)

        label = self._render_pixel_text(self.font, text, self.HUD_DIM)
        label_rect = label.get_rect(center=pygame.Rect(rect).center)
        self.screen.blit(label, label_rect)

    def _draw_nine_slice(self, surface, sprite, rect, src_corner_px):
        rect = pygame.Rect(rect)
        dst_corner_px = max(1, int(src_corner_px * self.ui_scale))
        self._draw_nine_slice_scaled(surface, sprite, rect, src_corner_px, dst_corner_px)

    def _draw_nine_slice_scaled(self, surface, sprite, rect, src_corner, dst_corner):
        rect = pygame.Rect(rect)
        dst_corner = min(dst_corner, rect.width // 2, rect.height // 2)
        src_w, src_h = sprite.get_size()
        src_mid_w = src_w - src_corner * 2
        src_mid_h = src_h - src_corner * 2
        dst_mid_w = max(0, rect.width - dst_corner * 2)
        dst_mid_h = max(0, rect.height - dst_corner * 2)

        def blit_slice(src_rect, dst_rect):
            if dst_rect.width <= 0 or dst_rect.height <= 0:
                return
            tile = sprite.subsurface(src_rect)
            surface.blit(pygame.transform.scale(tile, dst_rect.size), dst_rect)

        slices = [
            ((0, 0, src_corner, src_corner),
             (rect.left, rect.top, dst_corner, dst_corner)),
            ((src_w - src_corner, 0, src_corner, src_corner),
             (rect.right - dst_corner, rect.top, dst_corner, dst_corner)),
            ((0, src_h - src_corner, src_corner, src_corner),
             (rect.left, rect.bottom - dst_corner, dst_corner, dst_corner)),
            ((src_w - src_corner, src_h - src_corner, src_corner, src_corner),
             (rect.right - dst_corner, rect.bottom - dst_corner, dst_corner, dst_corner)),
            ((src_corner, 0, src_mid_w, src_corner),
             (rect.left + dst_corner, rect.top, dst_mid_w, dst_corner)),
            ((src_corner, src_h - src_corner, src_mid_w, src_corner),
             (rect.left + dst_corner, rect.bottom - dst_corner, dst_mid_w, dst_corner)),
            ((0, src_corner, src_corner, src_mid_h),
             (rect.left, rect.top + dst_corner, dst_corner, dst_mid_h)),
            ((src_w - src_corner, src_corner, src_corner, src_mid_h),
             (rect.right - dst_corner, rect.top + dst_corner, dst_corner, dst_mid_h)),
            ((src_corner, src_corner, src_mid_w, src_mid_h),
             (rect.left + dst_corner, rect.top + dst_corner, dst_mid_w, dst_mid_h)),
        ]
        for src_rect, dst_rect in slices:
            blit_slice(pygame.Rect(src_rect), pygame.Rect(dst_rect))

    def _draw_hud(self, state, info, gi):
        vspd = -state.vy
        hspd = state.vx
        thr = info.get("throttle", 0.0)
        wind = info.get("wind", 0.0)

        text_specs = [
            (self.font_lg, "ROCKET SIM", self.HUD, 6),
            (self.font, f"GEN {gi.get('gen','?')}/{gi.get('total_gen','?')}", self.HUD_DIM, 4),
            (self.font, f"EVAL {gi.get('fitness',0):.1f}", self.HUD_DIM, 4),
            (self.font, f"WIN {gi.get('success_rate',0):.0%}", self.HUD_DIM, 8),
            (self.font, f"ALT   {state.y:6.1f} M", self.HUD, 6),
            (self.font, f"VSPD  {vspd:+6.1f}", self.HUD, 6),
            (self.font, f"HSPD  {hspd:+6.1f}", self.HUD, 6),
            (self.font, f"TILT  {np.degrees(state.theta):+6.1f}", self.HUD, 6),
            (self.font, f"WIND  {wind:+6.1f}", self.HUD, 6),
            (self.font, f"ZOOM  {self.camera_zoom:.2f}X", self.HUD, 6),
            (self.font, "WASD PAN  +/- ZOOM", self.HUD_DIM, 6),
        ]
        rendered_text = [
            (self._render_pixel_text(font, text, color), gap)
            for font, text, color, gap in text_specs
        ]

        panel = pygame.Rect(self.HUD_PANEL_MARGIN, self.HUD_PANEL_MARGIN,
                            self.HUD_PANEL_WIDTH, 0)
        x = panel.left + self.HUD_PAD
        y = panel.top + self.HUD_PAD
        content_w = panel.width - self.HUD_PAD * 2

        # Measure layout once so the panel can sit behind the full overlay.
        panel.height = (
            self.HUD_PAD * 2
            + sum(surf.get_height() + gap for surf, gap in rendered_text)
            + 42
        )
        self._draw_ui_panel(self.screen, panel)

        for surf, gap in rendered_text:
            self.screen.blit(surf, (x, y))
            y += surf.get_height() + gap
        self._draw_button(pygame.Rect(x, y, content_w, 42), "SPACE SKIP")

        meter_y = panel.bottom + 12
        label_gap = 6
        meter_gap = 10
        for label, value, color, dark_color in [
            (f"FUEL {state.fuel:.0%}", state.fuel,
             self.METER_FUEL, self.METER_FUEL_DARK),
            (f"THRUST {thr:.0%}", thr,
             self.METER_THRUST, self.METER_THRUST_DARK),
        ]:
            label_surf = self._render_pixel_text(self.font, label, self.HUD)
            self.screen.blit(label_surf, (panel.left + self.HUD_PAD, meter_y))
            meter_y += label_surf.get_height() + label_gap
            meter_rect = pygame.Rect(panel.left, meter_y, panel.width, 56)
            self._draw_meter(meter_rect, value, color, dark_color)
            meter_y = meter_rect.bottom + meter_gap
