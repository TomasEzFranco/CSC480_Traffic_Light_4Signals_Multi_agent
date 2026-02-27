# simulation.py
# =============================================================================
# CSC 480 — Advanced Traffic Intersection Simulation
# =============================================================================

import random
import time
import threading
import sys
import os
import csv
import math
import argparse
import collections
from datetime import datetime

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="CSC 480 Traffic Simulation")
parser.add_argument("--headless",   action="store_true")
parser.add_argument("--mode",       default="adaptive",
                    choices=["fixed", "adaptive", "greedy", "random"])
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--duration",   type=int,   default=0)
parser.add_argument("--spawn-rate", type=float, default=1.0)
parser.add_argument("--log",        default="metrics.csv")
parser.add_argument("--no-log",     action="store_true")
args = parser.parse_args()

HEADLESS = args.headless
if HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame

# =============================================================================
# CONFIG
# =============================================================================
class Config:
    TILE_W           = 1400
    TILE_H           = 800
    GRID_COLS        = 2
    GRID_ROWS        = 2
    NO_INTERSECTIONS = 4

    # Signal timing
    DEFAULT_GREEN    = {0: 10, 1: 10, 2: 10, 3: 10}
    DEFAULT_RED      = 150
    DEFAULT_YELLOW   = 5
    MIN_GREEN        = 5
    MAX_GREEN        = 30

    # Physics
    SPEEDS           = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5}
    ACCEL            = 0.12
    DECEL            = 0.20

    # Spawning
    SPAWN_INTERVAL   = 1.0 / max(args.spawn_rate, 0.1)
    DIRECTION_DIST   = [25, 50, 75, 100]

    # Gaps
    STOPPING_GAP     = 15
    MOVING_GAP       = 15

    # Camera
    CAMERA_SPEED     = 30
    CAMERA_SMOOTH    = 0.15

    # HUD
    HUD_WIDTH        = 310
    FPS              = 60

    # Metrics
    METRIC_INTERVAL  = 5.0
    LOG_FILE         = args.log
    DISABLE_LOG      = args.no_log

    # Run params
    SEED             = args.seed
    DURATION         = args.duration
    CONTROL_MODE     = args.mode
    SPAWN_RATE       = args.spawn_rate

    # Visuals
    SHOW_HEATMAP     = True
    SHOW_VECTORS     = False
    SHOW_GRID        = True
    PARTICLE_EFFECTS = True

    # Layout
    OFFSETS = {
        0: (0,      0),
        1: (TILE_W, 0),
        2: (0,      TILE_H),
        3: (TILE_W, TILE_H),
    }

    VEHICLE_TYPES  = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
    DIRECTION_NUMS = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
    NO_OF_SIGNALS  = 4

    # -----------------------------------------------------------------------
    # DIRECTION → SIGNAL INDEX mapping
    # This is the critical constant that was missing.
    # Signal index 0 controls 'right', 1 controls 'down',
    # 2 controls 'left', 3 controls 'up'.
    # A vehicle moving 'right' checks if cur_green == DIRECTION_TO_SIGNAL['right']
    # -----------------------------------------------------------------------
    DIRECTION_TO_SIGNAL = {
        'right': 0,
        'down':  1,
        'left':  2,
        'up':    3,
    }

    # Spawn coords (base tile, per lane index 0/1/2)
    BASE_X = {
        'right': [0,    0,    0   ],
        'down':  [755,  727,  697 ],
        'left':  [1400, 1400, 1400],
        'up':    [602,  627,  657 ],
    }
    BASE_Y = {
        'right': [348, 370, 398],
        'down':  [0,   0,   0  ],
        'left':  [498, 466, 436],
        'up':    [800, 800, 800],
    }

    BASE_SIGNAL_COORDS       = [(530,230),(810,230),(810,570),(530,570)]
    BASE_SIGNAL_TIMER_COORDS = [(530,210),(810,210),(810,550),(530,550)]

    BASE_STOP_LINES   = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
    BASE_DEFAULT_STOP = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

    # Adaptive agent weights
    ADAPTIVE_W_QUEUE      = 1.0
    ADAPTIVE_W_WAIT       = 0.8
    ADAPTIVE_W_STARVATION = 2.5
    ADAPTIVE_STARVATION_T = 45

C = Config()
random.seed(C.SEED)

# =============================================================================
# IMAGE CACHE
# Loads images once. In headless mode skips convert_alpha() which requires
# an active display — the root cause of the "No video mode has been set" crash.
# =============================================================================
_image_cache: dict = {}

def load_image(path: str) -> pygame.Surface:
    """
    Load image with caching.
    Uses convert_alpha() only when a display is available (non-headless).
    In headless mode returns a plain Surface so pygame.image.load() works
    without an active video mode.
    """
    if path in _image_cache:
        return _image_cache[path]

    surf = pygame.image.load(path)
    if not HEADLESS:
        surf = surf.convert_alpha()

    _image_cache[path] = surf
    return surf

# =============================================================================
# METRICS ENGINE
# =============================================================================
class MetricsEngine:
    def __init__(self):
        self.start_time         = time.time()
        self.last_log_time      = self.start_time
        self.total_crossed      = [0] * C.NO_INTERSECTIONS
        self.total_spawned      = [0] * C.NO_INTERSECTIONS
        self.wait_samples       = [collections.deque(maxlen=500)
                                   for _ in range(C.NO_INTERSECTIONS)]
        self.queue_history      = [collections.deque(maxlen=300)
                                   for _ in range(C.NO_INTERSECTIONS)]
        self.vehicle_entry_time = {}
        self._csv_file          = None
        self._csv_writer        = None
        if not C.DISABLE_LOG:
            self._init_csv()

    def _init_csv(self):
        fname        = C.LOG_FILE
        write_header = not os.path.exists(fname)
        self._csv_file   = open(fname, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if write_header:
            self._csv_writer.writerow([
                "timestamp", "elapsed_s", "mode", "seed", "spawn_rate", "iid",
                "avg_wait_s", "p95_wait_s", "max_wait_s",
                "avg_queue", "max_queue",
                "throughput_per_min", "fairness_index",
                "total_crossed", "total_spawned",
            ])

    def vehicle_spawned(self, vehicle):
        self.vehicle_entry_time[vehicle.uid] = time.time()
        self.total_spawned[vehicle.iid] += 1

    def vehicle_crossed(self, vehicle):
        iid = vehicle.iid
        self.total_crossed[iid] += 1
        entry = self.vehicle_entry_time.pop(vehicle.uid, None)
        if entry is not None:
            self.wait_samples[iid].append(time.time() - entry)

    def record_queue(self, iid, q):
        self.queue_history[iid].append(q)

    def get_avg_wait(self, iid):
        s = self.wait_samples[iid]
        return sum(s) / len(s) if s else 0.0

    def get_p95_wait(self, iid):
        s = sorted(self.wait_samples[iid])
        if not s:
            return 0.0
        return s[min(int(0.95 * len(s)), len(s) - 1)]

    def get_max_wait(self, iid):
        return max(self.wait_samples[iid], default=0.0)

    def get_avg_queue(self, iid):
        h = self.queue_history[iid]
        return sum(h) / len(h) if h else 0.0

    def get_max_queue(self, iid):
        return max(self.queue_history[iid], default=0)

    def get_throughput(self, iid):
        elapsed = max(time.time() - self.start_time, 1.0)
        return self.total_crossed[iid] / elapsed * 60.0

    def get_fairness_index(self, iid, vehicles_state):
        qs = [
            sum(len(vehicles_state[iid][d][lane]) for lane in range(3))
            for d in C.DIRECTION_NUMS.values()
        ]
        total = sum(qs)
        if total == 0:
            return 1.0
        n   = len(qs)
        num = total ** 2
        den = n * sum(q * q for q in qs)
        return num / den if den > 0 else 1.0

    def maybe_log(self, vehicles_state):
        if C.DISABLE_LOG or self._csv_writer is None:
            return
        now = time.time()
        if now - self.last_log_time < C.METRIC_INTERVAL:
            return
        self.last_log_time = now
        elapsed = now - self.start_time
        ts      = datetime.utcnow().isoformat()
        for iid in range(C.NO_INTERSECTIONS):
            qs = [
                sum(len(vehicles_state[iid][d][lane]) for lane in range(3))
                for d in C.DIRECTION_NUMS.values()
            ]
            self._csv_writer.writerow([
                ts, f"{elapsed:.1f}", C.CONTROL_MODE, C.SEED, C.SPAWN_RATE, iid,
                f"{self.get_avg_wait(iid):.3f}",
                f"{self.get_p95_wait(iid):.3f}",
                f"{self.get_max_wait(iid):.3f}",
                f"{self.get_avg_queue(iid):.3f}",
                f"{self.get_max_queue(iid)}",
                f"{self.get_throughput(iid):.3f}",
                f"{self.get_fairness_index(iid, vehicles_state):.4f}",
                self.total_crossed[iid],
                self.total_spawned[iid],
            ])
        self._csv_file.flush()

    def snapshot(self, iid, vehicles_state):
        qs = [
            sum(len(vehicles_state[iid][d][lane]) for lane in range(3))
            for d in C.DIRECTION_NUMS.values()
        ]
        return {
            "avg_wait":   self.get_avg_wait(iid),
            "p95_wait":   self.get_p95_wait(iid),
            "max_wait":   self.get_max_wait(iid),
            "throughput": self.get_throughput(iid),
            "avg_queue":  self.get_avg_queue(iid),
            "max_queue":  self.get_max_queue(iid),
            "queues":     qs,
            "crossed":    self.total_crossed[iid],
            "spawned":    self.total_spawned[iid],
            "fairness":   self.get_fairness_index(iid, vehicles_state),
        }

    def close(self):
        if self._csv_file:
            self._csv_file.close()

metrics = MetricsEngine()

# =============================================================================
# TRAFFIC SIGNAL
# =============================================================================
class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red        = red
        self.yellow     = yellow
        self.green      = green
        self.signalText = ""

# =============================================================================
# SIGNAL CONTROLLERS
# =============================================================================
class SignalController:
    """
    Base class / RL Agent API.
    Subclass and override next_green_duration() and choose_next_phase().
    """
    def __init__(self, iid):
        self.iid = iid

    def get_state(self, signals_list, cur_green, cur_yellow, vehicles_state):
        iid    = self.iid
        queues = []
        for dn in range(4):
            d = C.DIRECTION_NUMS[dn]
            q = sum(1 for lane in range(3)
                    for v in vehicles_state[iid][d][lane]
                    if v.crossed == 0)
            queues.append(q)
        sigs = signals_list[iid]
        tr   = (sigs[cur_green[iid]].green if not cur_yellow[iid]
                else sigs[cur_green[iid]].yellow)
        return {
            "iid":            iid,
            "phase":          cur_green[iid],
            "time_remaining": tr,
            "queue_lengths":  queues,
            "yellow_active":  bool(cur_yellow[iid]),
        }

    def compute_reward(self, state_before, state_after):
        q_b = sum(state_before["queue_lengths"])
        q_a = sum(state_after["queue_lengths"])
        avg = q_a / 4.0
        var = sum((q - avg) ** 2 for q in state_after["queue_lengths"]) / 4.0
        return -(q_a + 0.1 * q_b + 0.05 * var)

    def next_green_duration(self, signals_list, cur_green, vehicles_state):
        return C.DEFAULT_GREEN[cur_green[self.iid]]

    def choose_next_phase(self, vehicles_state, cur_green):
        # Default: round-robin
        return (cur_green[self.iid] + 1) % C.NO_OF_SIGNALS


class FixedTimeController(SignalController):
    """
    PAIR 1 — Baseline.
    Fixed cycle, fixed duration. Every direction gets exactly the same
    green time in strict round-robin order regardless of traffic load.
    """
    def next_green_duration(self, signals_list, cur_green, vehicles_state):
        return C.DEFAULT_GREEN[cur_green[self.iid]]

    def choose_next_phase(self, vehicles_state, cur_green):
        return (cur_green[self.iid] + 1) % C.NO_OF_SIGNALS


class GreedyController(SignalController):
    """
    PAIR 2 — Greedy / Rule-Based.
    Always grants green to the direction with the most waiting vehicles.
    Duration scales with queue depth.
    AI class: goal-based / simple-reflex agent.
    """
    def choose_next_phase(self, vehicles_state, cur_green):
        iid    = self.iid
        queues = [
            sum(len(vehicles_state[iid][C.DIRECTION_NUMS[dn]][lane])
                for lane in range(3))
            for dn in range(4)
        ]
        return max(range(4), key=lambda i: queues[i])

    def next_green_duration(self, signals_list, cur_green, vehicles_state):
        iid = self.iid
        d   = C.DIRECTION_NUMS[cur_green[iid]]
        q   = sum(len(vehicles_state[iid][d][lane]) for lane in range(3))
        return int(max(C.MIN_GREEN, min(C.MAX_GREEN, C.MIN_GREEN + q * 1.5)))


class AdaptiveController(SignalController):
    """
    PAIR 3 — Utility-Based Agent.

    Utility per direction:
        U(d) = W_QUEUE      * queue_length(d)
             + W_WAIT       * avg_accumulated_wait(d)
             + W_STARVATION * max(0, time_since_green(d) - THRESHOLD)

    Highest U wins the next green phase.
    Duration scales with winning direction's queue.
    AI class: utility-based agent with explicit multi-component objective.
    """
    def __init__(self, iid):
        super().__init__(iid)
        self._last_green_time = {dn: time.time() for dn in range(4)}

    def _utility(self, vehicles_state, cur_green):
        iid  = self.iid
        now  = time.time()
        util = {}
        for dn in range(4):
            d = C.DIRECTION_NUMS[dn]
            q = sum(len(vehicles_state[iid][d][lane]) for lane in range(3))
            waits = [
                now - v.wait_start
                for lane in range(3)
                for v in vehicles_state[iid][d][lane]
                if v.crossed == 0 and v.waiting and v.wait_start
            ]
            avg_wait   = sum(waits) / len(waits) if waits else 0.0
            starvation = max(
                0.0,
                (now - self._last_green_time.get(dn, now))
                - C.ADAPTIVE_STARVATION_T
            )
            util[dn] = (
                C.ADAPTIVE_W_QUEUE      * q        +
                C.ADAPTIVE_W_WAIT       * avg_wait +
                C.ADAPTIVE_W_STARVATION * starvation
            )
        return util

    def choose_next_phase(self, vehicles_state, cur_green):
        self._last_green_time[cur_green[self.iid]] = time.time()
        return max(self._utility(vehicles_state, cur_green),
                   key=self._utility(vehicles_state, cur_green).get)

    def next_green_duration(self, signals_list, cur_green, vehicles_state):
        iid = self.iid
        d   = C.DIRECTION_NUMS[cur_green[iid]]
        q   = sum(len(vehicles_state[iid][d][lane]) for lane in range(3))
        return int(max(C.MIN_GREEN, min(C.MAX_GREEN, C.MIN_GREEN + q * 1.8)))


class RandomController(SignalController):
    """Null baseline — random durations AND random phase order."""
    def next_green_duration(self, signals_list, cur_green, vehicles_state):
        return random.randint(C.MIN_GREEN, C.MAX_GREEN)

    def choose_next_phase(self, vehicles_state, cur_green):
        return random.randint(0, 3)


CONTROLLER_MAP = {
    "fixed":    FixedTimeController,
    "adaptive": AdaptiveController,
    "greedy":   GreedyController,
    "random":   RandomController,
}

# =============================================================================
# PARTICLES
# =============================================================================
class Particle:
    __slots__ = ["x", "y", "vx", "vy", "life", "max_life", "color", "size"]
    _OX = {'right': -1, 'left': 1, 'down': 0, 'up': 0}
    _OY = {'right': 0,  'left': 0, 'down': -1, 'up': 1}

    def __init__(self, x, y, direction):
        self.x        = x + random.uniform(-3, 3)
        self.y        = y + random.uniform(-3, 3)
        self.vx       = self._OX[direction] * random.uniform(0.4, 1.4) + random.uniform(-0.3, 0.3)
        self.vy       = self._OY[direction] * random.uniform(0.4, 1.4) + random.uniform(-0.3, 0.3)
        self.max_life = random.randint(18, 42)
        self.life     = self.max_life
        g             = random.randint(130, 195)
        self.color    = (g, g, g)
        self.size     = random.uniform(1.5, 3.5)

    def update(self):
        self.x    += self.vx
        self.y    += self.vy
        self.life -= 1
        self.size  = max(0.0, self.size - 0.06)

    @property
    def alive(self):
        return self.life > 0

    def draw(self, surface):
        if self.size >= 1:
            pygame.draw.circle(surface, self.color,
                               (int(self.x), int(self.y)), int(self.size))

_particle_pool: list = []

# =============================================================================
# GLOBAL VEHICLE STATE  (initialised in SimState.__init__)
# =============================================================================
_spawn_x  = None
_spawn_y  = None
_vehicles = None

# =============================================================================
# COORDINATE HELPERS
# =============================================================================
def stop_line(iid: int, direction: str) -> float:
    ox, oy = C.OFFSETS[iid]
    base   = C.BASE_STOP_LINES[direction]
    return base + (ox if direction in ('right', 'left') else oy)

def default_stop_pos(iid: int, direction: str) -> float:
    ox, oy = C.OFFSETS[iid]
    base   = C.BASE_DEFAULT_STOP[direction]
    return base + (ox if direction in ('right', 'left') else oy)

# =============================================================================
# VEHICLE
# =============================================================================
class Vehicle(pygame.sprite.Sprite):
    _id_counter = 0

    def __init__(self, iid, lane, vehicleClass, direction_number, direction,
                 sim_group):
        pygame.sprite.Sprite.__init__(self)

        Vehicle._id_counter += 1
        self.uid              = Vehicle._id_counter
        self.iid              = iid
        self.lane             = lane
        self.vehicleClass     = vehicleClass
        self.speed            = C.SPEEDS[vehicleClass]
        self.current_speed    = 0.0
        self.direction_number = direction_number
        self.direction        = direction
        self.crossed          = 0
        self.waiting          = False
        self.wait_start       = None
        self.total_wait_s     = 0.0
        self._particle_timer  = 0
        self._offscreen       = False

        self.x = _spawn_x[iid][direction][lane]
        self.y = _spawn_y[iid][direction][lane]

        # ---- Image loading — headless-safe ----
        self.image = load_image(f"images/{direction}/{vehicleClass}.png")
        self.rect  = self.image.get_rect()

        # Register
        _vehicles[iid][direction][lane].append(self)
        self.index = len(_vehicles[iid][direction][lane]) - 1

        # ---- Stop position ----
        if (self.index > 0 and
                _vehicles[iid][direction][lane][self.index - 1].crossed == 0):
            prev = _vehicles[iid][direction][lane][self.index - 1]
            if direction == 'right':
                self.stop = prev.stop - prev.rect.width  - C.STOPPING_GAP
            elif direction == 'left':
                self.stop = prev.stop + prev.rect.width  + C.STOPPING_GAP
            elif direction == 'down':
                self.stop = prev.stop - prev.rect.height - C.STOPPING_GAP
            elif direction == 'up':
                self.stop = prev.stop + prev.rect.height + C.STOPPING_GAP
        else:
            self.stop = default_stop_pos(iid, direction)

        # ---- Advance spawn stack ----
        if direction == 'right':
            _spawn_x[iid][direction][lane] -= self.rect.width  + C.STOPPING_GAP
        elif direction == 'left':
            _spawn_x[iid][direction][lane] += self.rect.width  + C.STOPPING_GAP
        elif direction == 'down':
            _spawn_y[iid][direction][lane] -= self.rect.height + C.STOPPING_GAP
        elif direction == 'up':
            _spawn_y[iid][direction][lane] += self.rect.height + C.STOPPING_GAP

        metrics.vehicle_spawned(self)
        sim_group.add(self)

    # ------------------------------------------------------------------
    def move(self, cur_green, cur_yellow):
        """
        Movement logic.

        KEY FIX — red-light enforcement:
        A vehicle's 'direction' maps to a fixed signal index via
        C.DIRECTION_TO_SIGNAL.  We compare cur_green[iid] against that
        index — NOT against direction_number which can be anything.

        This means:
          - right  traffic checks signal slot 0
          - down   traffic checks signal slot 1
          - left   traffic checks signal slot 2
          - up     traffic checks signal slot 3

        This is always true regardless of which phase the controller
        chose to run last.
        """
        iid = self.iid
        d   = self.direction
        cg  = cur_green[iid]
        cy  = cur_yellow[iid]
        w   = self.rect.width
        h   = self.rect.height

        # My signal slot
        my_signal = C.DIRECTION_TO_SIGNAL[d]

        # ---- 1. Crossing detection ----
        sl = stop_line(iid, d)
        if self.crossed == 0:
            hit = (
                (d == 'right' and self.x + w > sl) or
                (d == 'left'  and self.x       < sl) or
                (d == 'down'  and self.y + h   > sl) or
                (d == 'up'    and self.y        < sl)
            )
            if hit:
                self.crossed = 1
                metrics.vehicle_crossed(self)

        # ---- 2. Is it my green? ----
        # Green only when:  the current green phase matches MY signal slot
        #                   AND we are not in yellow
        my_green = (cg == my_signal) and (cy == 0)

        # ---- 3. Gap to vehicle ahead ----
        gap_ok = True
        if self.index > 0:
            prev = _vehicles[iid][d][self.lane][self.index - 1]
            if d == 'right':
                gap_ok = self.x + w < prev.x - C.MOVING_GAP
            elif d == 'left':
                gap_ok = self.x > prev.x + prev.rect.width + C.MOVING_GAP
            elif d == 'down':
                gap_ok = self.y + h < prev.y - C.MOVING_GAP
            elif d == 'up':
                gap_ok = self.y > prev.y + prev.rect.height + C.MOVING_GAP

        # ---- 4. Should this vehicle move? ----
        # Mirrors the original proven logic exactly:
        #   - if not yet at stop line → keep moving (approach)
        #   - if at/past stop line AND crossed → keep moving (already through)
        #   - if at/past stop line AND green  → keep moving (green light)
        #   - otherwise → stop
        if d == 'right':
            at_stop = self.x + w >= self.stop
            can_move = (not at_stop or self.crossed == 1 or my_green) and gap_ok
        elif d == 'down':
            at_stop  = self.y + h >= self.stop
            can_move = (not at_stop or self.crossed == 1 or my_green) and gap_ok
        elif d == 'left':
            at_stop  = self.x <= self.stop
            can_move = (not at_stop or self.crossed == 1 or my_green) and gap_ok
        elif d == 'up':
            at_stop  = self.y <= self.stop
            can_move = (not at_stop or self.crossed == 1 or my_green) and gap_ok
        else:
            can_move = False

        # ---- 5. Smooth speed ----
        target = self.speed if can_move else 0.0
        if self.current_speed < target:
            self.current_speed = min(self.current_speed + C.ACCEL, target)
        else:
            self.current_speed = max(self.current_speed - C.DECEL, target)

        # ---- 6. Wait tracking ----
        if self.current_speed < 0.05:
            if not self.waiting:
                self.waiting    = True
                self.wait_start = time.time()
        else:
            if self.waiting:
                self.waiting = False
                if self.wait_start is not None:
                    self.total_wait_s += time.time() - self.wait_start
                self.wait_start = None

        # ---- 7. Apply movement ----
        if d == 'right': self.x += self.current_speed
        elif d == 'left': self.x -= self.current_speed
        elif d == 'down': self.y += self.current_speed
        elif d == 'up':   self.y -= self.current_speed

        # ---- 8. Particles ----
        if C.PARTICLE_EFFECTS and self.current_speed > 0.5:
            self._particle_timer += 1
            if self._particle_timer % 8 == 0:
                _particle_pool.append(Particle(self.x, self.y, d))

        # ---- 9. Off-screen cull ----
        # Use a generous margin (200px) so vehicles that are still
        # partially on the far side of a tile boundary don't get culled.
        ox, oy = C.OFFSETS[iid]
        margin = 200
        if (self.x < ox - margin or self.x > ox + C.TILE_W + margin or
                self.y < oy - margin or self.y > oy + C.TILE_H + margin):
            self._offscreen = True

# =============================================================================
# SIMULATION STATE
# =============================================================================
class SimState:
    def __init__(self):
        global _spawn_x, _spawn_y, _vehicles

        self.sim_group   = pygame.sprite.Group()
        self.signals     = [[] for _ in range(C.NO_INTERSECTIONS)]

        # -----------------------------------------------------------------------
        # cur_green[iid] stores the SIGNAL SLOT index (0-3) that is currently
        # green. Signal slot 0 = right, 1 = down, 2 = left, 3 = up.
        # We initialise slot 0 (right) as green at every intersection.
        # -----------------------------------------------------------------------
        self.cur_green   = [0] * C.NO_INTERSECTIONS
        self.next_green  = [1] * C.NO_INTERSECTIONS
        self.cur_yellow  = [0] * C.NO_INTERSECTIONS

        ctrl_cls = CONTROLLER_MAP.get(C.CONTROL_MODE, AdaptiveController)
        self.controllers = [ctrl_cls(iid) for iid in range(C.NO_INTERSECTIONS)]

        # ---- Spawn stacks ----
        _spawn_x  = []
        _spawn_y  = []
        _vehicles = []
        for iid in range(C.NO_INTERSECTIONS):
            ox, oy = C.OFFSETS[iid]
            sx, sy = {}, {}
            for d in C.DIRECTION_NUMS.values():
                sx[d] = [C.BASE_X[d][l] + ox for l in range(3)]
                sy[d] = [C.BASE_Y[d][l] + oy for l in range(3)]
            _spawn_x.append(sx)
            _spawn_y.append(sy)
            _vehicles.append({d: {0: [], 1: [], 2: []}
                               for d in C.DIRECTION_NUMS.values()})

        self.vehicles    = _vehicles
        self.start_time  = time.time()
        self.paused      = False
        self.frame_count = 0

        # Signal threads
        for iid in range(C.NO_INTERSECTIONS):
            threading.Thread(
                target=self._init_signals, args=(iid,),
                daemon=True, name=f"sig_{iid}"
            ).start()

        time.sleep(0.4)   # let signals initialise before vehicles spawn

        # Vehicle generator thread
        threading.Thread(
            target=self._generate_vehicles,
            daemon=True, name="gen"
        ).start()

    # ------------------------------------------------------------------
    # Signal management
    # ------------------------------------------------------------------
    def _init_signals(self, iid):
        ts1 = TrafficSignal(0,             C.DEFAULT_YELLOW, C.DEFAULT_GREEN[0])
        ts2 = TrafficSignal(
            ts1.red + ts1.yellow + ts1.green,
            C.DEFAULT_YELLOW, C.DEFAULT_GREEN[1]
        )
        ts3 = TrafficSignal(C.DEFAULT_RED, C.DEFAULT_YELLOW, C.DEFAULT_GREEN[2])
        ts4 = TrafficSignal(C.DEFAULT_RED, C.DEFAULT_YELLOW, C.DEFAULT_GREEN[3])
        self.signals[iid] = [ts1, ts2, ts3, ts4]
        self._signal_loop(iid)

    def _signal_loop(self, iid):
        """
        Signal timing loop for one intersection.

        cur_green[iid] is the index into self.signals[iid] AND the index
        into DIRECTION_TO_SIGNAL — they use the same numbering:
            0 = right, 1 = down, 2 = left, 3 = up

        This guarantees vehicles always know exactly which signal is theirs.
        """
        while True:
            ctrl = self.controllers[iid]

            # Let controller set green duration
            dur = ctrl.next_green_duration(
                self.signals, self.cur_green, self.vehicles
            )
            self.signals[iid][self.cur_green[iid]].green = dur

            # Green countdown
            while self.signals[iid][self.cur_green[iid]].green > 0:
                if not self.paused:
                    self._tick(iid)
                time.sleep(1)

            # Controller picks next phase
            self.next_green[iid] = ctrl.choose_next_phase(
                self.vehicles, self.cur_green
            )

            # Yellow phase
            self.cur_yellow[iid] = 1
            d_cur = C.DIRECTION_NUMS[self.cur_green[iid]]
            for lane in range(3):
                for v in self.vehicles[iid][d_cur][lane]:
                    v.stop = default_stop_pos(iid, d_cur)

            while self.signals[iid][self.cur_green[iid]].yellow > 0:
                if not self.paused:
                    self._tick(iid)
                time.sleep(1)

            self.cur_yellow[iid] = 0

            # Reset outgoing signal timers
            cg = self.cur_green[iid]
            self.signals[iid][cg].green  = C.DEFAULT_GREEN[cg]
            self.signals[iid][cg].yellow = C.DEFAULT_YELLOW
            self.signals[iid][cg].red    = C.DEFAULT_RED

            # Advance to next phase
            self.cur_green[iid]  = self.next_green[iid]
            self.next_green[iid] = (self.cur_green[iid] + 1) % C.NO_OF_SIGNALS

            ng     = self.next_green[iid]
            cg_new = self.cur_green[iid]
            self.signals[iid][ng].red = (
                self.signals[iid][cg_new].yellow +
                self.signals[iid][cg_new].green
            )

    def _tick(self, iid):
        for i in range(C.NO_OF_SIGNALS):
            if i == self.cur_green[iid]:
                if self.cur_yellow[iid]:
                    self.signals[iid][i].yellow = max(
                        0, self.signals[iid][i].yellow - 1
                    )
                else:
                    self.signals[iid][i].green = max(
                        0, self.signals[iid][i].green - 1
                    )
            else:
                self.signals[iid][i].red = max(
                    0, self.signals[iid][i].red - 1
                )

    # ------------------------------------------------------------------
    # Vehicle generation
    # ------------------------------------------------------------------
    def _generate_vehicles(self):
        while True:
            if not self.paused:
                iid   = random.randint(0, C.NO_INTERSECTIONS - 1)
                vtype = random.randint(0, 3)
                lane  = random.randint(1, 2)
                t     = random.randint(0, 99)
                dist  = C.DIRECTION_DIST
                if   t < dist[0]: dn = 0
                elif t < dist[1]: dn = 1
                elif t < dist[2]: dn = 2
                else:             dn = 3
                Vehicle(iid, lane,
                        C.VEHICLE_TYPES[vtype],
                        dn, C.DIRECTION_NUMS[dn],
                        self.sim_group)
            time.sleep(C.SPAWN_INTERVAL)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def update(self):
        if self.paused:
            return

        self.frame_count += 1

        # Sample queue every 30 frames
        if self.frame_count % 30 == 0:
            for iid in range(C.NO_INTERSECTIONS):
                total_q = sum(
                    len(self.vehicles[iid][d][lane])
                    for d in C.DIRECTION_NUMS.values()
                    for lane in range(3)
                )
                metrics.record_queue(iid, total_q)

        # Cull and move
        for v in list(self.sim_group):
            if v._offscreen:
                v.kill()
            else:
                v.move(self.cur_green, self.cur_yellow)

        # Particles
        if C.PARTICLE_EFFECTS:
            for p in _particle_pool:
                p.update()
            _particle_pool[:] = [p for p in _particle_pool if p.alive]

        # CSV log
        metrics.maybe_log(self.vehicles)

        # Auto-exit
        if C.DURATION > 0 and time.time() - self.start_time >= C.DURATION:
            metrics.close()
            self._print_summary()
            print("\n[Simulation complete]")
            pygame.quit()
            sys.exit(0)

    def _print_summary(self):
        print("\n" + "=" * 65)
        print(f"  SUMMARY  mode={C.CONTROL_MODE}  "
              f"seed={C.SEED}  spawn={C.SPAWN_RATE}")
        print("=" * 65)
        for iid in range(C.NO_INTERSECTIONS):
            s = metrics.snapshot(iid, self.vehicles)
            print(f"\n  Intersection {iid}:")
            print(f"    Spawned      : {s['spawned']}")
            print(f"    Crossed      : {s['crossed']}")
            print(f"    Avg wait     : {s['avg_wait']:.2f}s")
            print(f"    P95 wait     : {s['p95_wait']:.2f}s")
            print(f"    Max wait     : {s['max_wait']:.2f}s")
            print(f"    Throughput   : {s['throughput']:.1f} veh/min")
            print(f"    Avg queue    : {s['avg_queue']:.1f}")
            print(f"    Fairness     : {s['fairness']:.4f}")
        print("=" * 65)

# =============================================================================
# RENDERER
# =============================================================================
class Renderer:
    BLACK  = (0,   0,   0)
    WHITE  = (255, 255, 255)
    RED    = (220, 50,  50)
    GREEN  = (50,  200, 80)
    YELLOW = (240, 200, 40)
    CYAN   = (50,  220, 220)
    ORANGE = (240, 140, 40)
    DARK   = (18,  18,  28)

    PHASE_COLORS = {
        'right': (50,  200, 240),
        'down':  (240, 120, 50),
        'left':  (200, 50,  200),
        'up':    (50,  240, 120),
    }
    MODE_COLORS = {
        "fixed":    (160, 160, 160),
        "greedy":   (80,  160, 255),
        "adaptive": (80,  230, 130),
        "random":   (230, 180, 60),
    }

    def __init__(self, state: SimState):
        self.state    = state
        self.SCREEN_W = C.TILE_W
        self.SCREEN_H = C.TILE_H
        self.WORLD_W  = C.TILE_W * C.GRID_COLS
        self.WORLD_H  = C.TILE_H * C.GRID_ROWS

        self.screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))
        pygame.display.set_caption(
            f"CSC 480  |  Agent: {C.CONTROL_MODE.upper()}  |  "
            "WASD=Camera  P=Pause  H=HUD  +/-=Spawn  1-4=Jump  F1=Stats"
        )
        self.world        = pygame.Surface((self.WORLD_W, self.WORLD_H))
        self.heatmap_surf = pygame.Surface(
            (self.WORLD_W, self.WORLD_H), pygame.SRCALPHA
        )
        self.hud_surf = pygame.Surface(
            (C.HUD_WIDTH, self.SCREEN_H), pygame.SRCALPHA
        )

        # Assets — safe to use convert_alpha() here because display exists
        self.bg     = load_image('images/intersection.png')
        self.sig_r  = load_image('images/signals/red.png')
        self.sig_y  = load_image('images/signals/yellow.png')
        self.sig_g  = load_image('images/signals/green.png')

        self.font_lg  = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_md  = pygame.font.SysFont("consolas", 16)
        self.font_sm  = pygame.font.SysFont("consolas", 13)
        self.font_xs  = pygame.font.SysFont("consolas", 11)
        self.font_sig = pygame.font.Font(None, 30)

        self.cam_x = self.cam_y = 0.0
        self.cam_tx = self.cam_ty = 0.0
        self.focused_iid = 0
        self.show_hud    = True

        self.minimap_rect = pygame.Rect(
            self.SCREEN_W - C.HUD_WIDTH,
            self.SCREEN_H - 145,
            C.HUD_WIDTH - 10, 135
        )
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    def _update_camera(self, keys):
        max_cx = self.WORLD_W - self.SCREEN_W
        max_cy = self.WORLD_H - self.SCREEN_H
        spd    = C.CAMERA_SPEED
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]: self.cam_tx = max(0,      self.cam_tx - spd)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: self.cam_tx = min(max_cx, self.cam_tx + spd)
        if keys[pygame.K_UP]    or keys[pygame.K_w]: self.cam_ty = max(0,      self.cam_ty - spd)
        if keys[pygame.K_DOWN]  or keys[pygame.K_s]: self.cam_ty = min(max_cy, self.cam_ty + spd)
        self.cam_x += (self.cam_tx - self.cam_x) * C.CAMERA_SMOOTH
        self.cam_y += (self.cam_ty - self.cam_y) * C.CAMERA_SMOOTH
        col = 1 if self.cam_x > C.TILE_W * 0.5 else 0
        row = 1 if self.cam_y > C.TILE_H * 0.5 else 0
        self.focused_iid = row * C.GRID_COLS + col

    # ------------------------------------------------------------------
    def _draw_world(self):
        self.world.fill(self.DARK)
        for iid in range(C.NO_INTERSECTIONS):
            ox, oy = C.OFFSETS[iid]
            self.world.blit(self.bg, (ox, oy))
        if C.SHOW_GRID:
            for col in range(1, C.GRID_COLS):
                px = col * C.TILE_W
                pygame.draw.line(self.world, (55,55,75), (px,0), (px, self.WORLD_H), 2)
            for row in range(1, C.GRID_ROWS):
                py = row * C.TILE_H
                pygame.draw.line(self.world, (55,55,75), (0,py), (self.WORLD_W,py), 2)
        if C.SHOW_HEATMAP:
            self._draw_heatmap()
        self._draw_signals()
        self._draw_vehicles()
        self._draw_labels()

    def _draw_heatmap(self):
        self.heatmap_surf.fill((0,0,0,0))
        for iid in range(C.NO_INTERSECTIONS):
            ox, oy = C.OFFSETS[iid]
            for dn, d in C.DIRECTION_NUMS.items():
                q = sum(len(self.state.vehicles[iid][d][lane]) for lane in range(3))
                if q == 0:
                    continue
                alpha = min(105, q * 15)
                col   = (*self.PHASE_COLORS[d], alpha)
                rects = {
                    'right': pygame.Rect(ox,      oy+340, 200, 60),
                    'down':  pygame.Rect(ox+680,  oy,      80, 200),
                    'left':  pygame.Rect(ox+1000, oy+440, 400,  60),
                    'up':    pygame.Rect(ox+560,  oy+540,  80, 260),
                }
                if d in rects:
                    pygame.draw.rect(self.heatmap_surf, col, rects[d])
        self.world.blit(self.heatmap_surf, (0,0))

    def _draw_signals(self):
        for iid in range(C.NO_INTERSECTIONS):
            if not self.state.signals[iid]:
                continue
            ox, oy = C.OFFSETS[iid]
            cg     = self.state.cur_green[iid]
            cy     = self.state.cur_yellow[iid]
            sigs   = self.state.signals[iid]
            for i in range(C.NO_OF_SIGNALS):
                sx = C.BASE_SIGNAL_COORDS[i][0]       + ox
                sy = C.BASE_SIGNAL_COORDS[i][1]       + oy
                tx = C.BASE_SIGNAL_TIMER_COORDS[i][0] + ox
                ty = C.BASE_SIGNAL_TIMER_COORDS[i][1] + oy
                if i == cg:
                    if cy:
                        sigs[i].signalText = sigs[i].yellow
                        self.world.blit(self.sig_y, (sx, sy))
                    else:
                        sigs[i].signalText = sigs[i].green
                        self.world.blit(self.sig_g, (sx, sy))
                else:
                    sigs[i].signalText = (sigs[i].red if sigs[i].red <= 10 else "---")
                    self.world.blit(self.sig_r, (sx, sy))
                txt = self.font_sig.render(str(sigs[i].signalText), True,
                                           self.WHITE, self.BLACK)
                self.world.blit(txt, (tx, ty))
                d_lbl = C.DIRECTION_NUMS[i][0].upper()
                lbl   = self.font_xs.render(d_lbl, True,
                                            self.PHASE_COLORS[C.DIRECTION_NUMS[i]])
                self.world.blit(lbl, (sx + 4, sy - 14))

    def _draw_vehicles(self):
        for v in self.state.sim_group:
            self.world.blit(v.image, (int(v.x), int(v.y)))
            if C.SHOW_VECTORS and v.current_speed < v.speed * 0.85:
                ratio = v.current_speed / max(v.speed, 0.01)
                bw    = int(v.rect.width * ratio)
                bc    = (int(220*(1-ratio)), int(220*ratio), 40)
                pygame.draw.rect(self.world, bc,
                                 (int(v.x), int(v.y)-5, bw, 3))

    def _draw_labels(self):
        mode_col = self.MODE_COLORS.get(C.CONTROL_MODE, self.WHITE)
        for iid in range(C.NO_INTERSECTIONS):
            ox, oy = C.OFFSETS[iid]
            lbl_txt = f" INT {iid}  {C.CONTROL_MODE.upper()} "
            lbl     = self.font_sm.render(lbl_txt, True, self.BLACK)
            pill    = pygame.Surface((lbl.get_width()+4, lbl.get_height()+4),
                                     pygame.SRCALPHA)
            pill.fill((*mode_col, 200))
            pill.blit(lbl, (2, 2))
            self.world.blit(pill, (ox+8, oy+8))

    def _draw_particles(self):
        for p in _particle_pool:
            p.draw(self.world)

    # ------------------------------------------------------------------
    def _draw_hud(self):
        if not self.show_hud:
            return
        W, H, pad = C.HUD_WIDTH, self.SCREEN_H, 9
        self.hud_surf.fill((0,0,0,0))
        bg = pygame.Surface((W, H), pygame.SRCALPHA)
        bg.fill((8, 8, 18, 210))
        self.hud_surf.blit(bg, (0,0))
        y = [pad]

        def write(text, font, color, indent=0):
            s = font.render(text, True, color)
            self.hud_surf.blit(s, (pad+indent, y[0]))
            y[0] += s.get_height() + 2

        def sep():
            pygame.draw.line(self.hud_surf, (50,50,70),
                             (pad, y[0]), (W-pad, y[0]), 1)
            y[0] += 5

        def bar(label, value, max_val, color, w=None):
            w = w or (W - pad*2)
            s = self.font_xs.render(label, True, (170,170,195))
            self.hud_surf.blit(s, (pad, y[0]))
            y[0] += s.get_height() + 1
            bw = int(w * min(value / max(max_val, 1), 1.0))
            pygame.draw.rect(self.hud_surf, (35,35,50),  (pad, y[0], w,  7))
            pygame.draw.rect(self.hud_surf, color,        (pad, y[0], bw, 7))
            y[0] += 10

        elapsed = int(time.time() - self.state.start_time)
        mm, ss  = divmod(elapsed, 60)
        fps     = self.clock.get_fps()
        mode_c  = self.MODE_COLORS.get(C.CONTROL_MODE, self.WHITE)

        write("CSC 480  TRAFFIC SIM", self.font_lg, self.CYAN)
        write(f"Agent: {C.CONTROL_MODE.upper()}", self.font_md, mode_c)
        write(f"Time {mm:02d}:{ss:02d}  FPS {fps:.0f}  "
              f"Veh {len(self.state.sim_group)}", self.font_xs, (170,210,170))
        write(f"Seed {C.SEED}  Spawn x{C.SPAWN_RATE:.1f}",
              self.font_xs, (150,150,190))
        sep()

        iid  = self.focused_iid
        snap = metrics.snapshot(iid, self.state.vehicles)
        write(f"INTERSECTION {iid}  [in view]", self.font_md, self.YELLOW)
        y[0] += 2
        write(f"Spawned   : {snap['spawned']}", self.font_xs, (200,200,200))
        write(f"Crossed   : {snap['crossed']}", self.font_xs, self.WHITE)
        write(f"Avg wait  : {snap['avg_wait']:.1f}s",  self.font_xs, self.ORANGE)
        write(f"P95 wait  : {snap['p95_wait']:.1f}s",  self.font_xs, self.ORANGE)
        write(f"Throughput: {snap['throughput']:.1f}/min", self.font_xs, self.GREEN)
        f_col = self.GREEN if snap['fairness'] > 0.75 else self.RED
        write(f"Fairness  : {snap['fairness']:.3f}", self.font_xs, f_col)
        y[0] += 3

        dir_labels = ['-> Right', 'v  Down', '<- Left', '^  Up']
        for i, (lbl, q) in enumerate(zip(dir_labels, snap['queues'])):
            bar(f"{lbl}: {q}", q, 20, self.PHASE_COLORS[C.DIRECTION_NUMS[i]])
        sep()

        cg   = self.state.cur_green[iid]
        cy   = self.state.cur_yellow[iid]
        sigs = self.state.signals[iid]
        if sigs:
            phase_d = C.DIRECTION_NUMS[cg]
            pcol    = self.YELLOW if cy else self.PHASE_COLORS[phase_d]
            pstr    = ("YELLOW  " if cy else "GREEN   ") + phase_d.upper()
            write(pstr, self.font_md, pcol)
            tr     = sigs[cg].green if not cy else sigs[cg].yellow
            max_tr = C.DEFAULT_GREEN[cg] if not cy else C.DEFAULT_YELLOW
            cx2 = W // 2
            r   = 26
            yc  = y[0] + r + 6
            pygame.draw.circle(self.hud_surf, (35,35,50), (cx2, yc), r)
            if max_tr > 0:
                self._arc(self.hud_surf, pcol, cx2, yc, r-4, 0,
                          360 * tr / max_tr)
            t_s = self.font_lg.render(str(tr), True, self.WHITE)
            self.hud_surf.blit(t_s, t_s.get_rect(center=(cx2, yc)))
            y[0] = yc + r + 8
        sep()

        write("ALL INTERSECTIONS", self.font_xs, (170,170,210))
        for i in range(C.NO_INTERSECTIONS):
            sn  = metrics.snapshot(i, self.state.vehicles)
            col = self.CYAN if i == iid else (150,150,175)
            write(f"  [{i}] cross:{sn['crossed']:3d}  "
                  f"W{sn['avg_wait']:4.1f}s  "
                  f"Q{sn['avg_queue']:4.1f}  "
                  f"F{sn['fairness']:.2f}",
                  self.font_xs, col)
        sep()

        write("CONTROLS", self.font_xs, (170,170,210))
        for key_s, act_s in [
            ("WASD/Arrows", "Pan camera"),
            ("1 2 3 4",     "Jump to intersection"),
            ("P",           "Pause/resume"),
            ("H",           "Toggle HUD"),
            ("V",           "Speed vectors"),
            ("+ / -",       "Spawn rate"),
            ("R",           "Reset camera"),
            ("F1",          "Print stats"),
        ]:
            ks  = self.font_xs.render(f"[{key_s}]", True, self.YELLOW)
            as_ = self.font_xs.render(act_s, True, (190,190,190))
            self.hud_surf.blit(ks,  (pad, y[0]))
            self.hud_surf.blit(as_, (pad + ks.get_width() + 4, y[0]))
            y[0] += ks.get_height() + 1

        self.screen.blit(self.hud_surf, (self.SCREEN_W - W, 0))

    def _arc(self, surf, color, cx, cy, r, start, end, width=4):
        steps = max(int(abs(end - start)), 1)
        for deg in range(0, steps, 3):
            a1 = math.radians(start + deg - 90)
            a2 = math.radians(start + deg + 3 - 90)
            pygame.draw.line(surf, color,
                (cx + int(r*math.cos(a1)), cy + int(r*math.sin(a1))),
                (cx + int(r*math.cos(a2)), cy + int(r*math.sin(a2))),
                width)

    def _draw_minimap(self):
        if not self.show_hud:
            return
        mr  = self.minimap_rect
        sx  = mr.width  / self.WORLD_W
        sy_ = mr.height / self.WORLD_H
        mm  = pygame.Surface((mr.width, mr.height))
        mm.fill((15,15,25))
        for iid in range(C.NO_INTERSECTIONS):
            ox, oy = C.OFFSETS[iid]
            pygame.draw.rect(mm, (45,45,65),
                (int(ox*sx), int(oy*sy_),
                 int(C.TILE_W*sx), int(C.TILE_H*sy_)))
            pygame.draw.rect(mm, (75,75,95),
                (int(ox*sx), int(oy*sy_),
                 int(C.TILE_W*sx), int(C.TILE_H*sy_)), 1)
        for v in self.state.sim_group:
            vx  = min(mr.width-1,  max(0, int(v.x * sx)))
            vy  = min(mr.height-1, max(0, int(v.y * sy_)))
            col = self.PHASE_COLORS.get(v.direction, (200,200,200))
            pygame.draw.circle(mm, col, (vx, vy), 1)
        cvx = int(self.cam_x * sx)
        cvy = int(self.cam_y * sy_)
        cvw = max(1, int(self.SCREEN_W * sx))
        cvh = max(1, int(self.SCREEN_H * sy_))
        pygame.draw.rect(mm, (255,255,80), (cvx, cvy, cvw, cvh), 1)
        pygame.draw.rect(mm, (100,100,140), (0,0,mr.width,mr.height), 1)
        mm.blit(self.font_xs.render("MINIMAP", True, (130,130,160)), (2,2))
        self.screen.blit(mm, mr.topleft)

    def _draw_paused(self):
        ov = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
        ov.fill((0,0,0,115))
        self.screen.blit(ov, (0,0))
        txt = self.font_lg.render("PAUSED  —  Press P to Resume",
                                  True, self.YELLOW)
        self.screen.blit(txt, txt.get_rect(
            center=(self.SCREEN_W//2, self.SCREEN_H//2)))

    def render(self, keys):
        self._update_camera(keys)
        self._draw_world()
        if C.PARTICLE_EFFECTS:
            self._draw_particles()
        self.screen.blit(self.world, (-int(self.cam_x), -int(self.cam_y)))
        if self.show_hud:
            self._draw_hud()
            self._draw_minimap()
        if self.state.paused:
            self._draw_paused()
        pygame.display.flip()
        self.clock.tick(C.FPS)

# =============================================================================
# MAIN
# =============================================================================
def main():
    pygame.init()

    # In headless mode we still need a minimal display init for pygame.font
    # and pygame.Surface to work, but we do NOT call set_mode().
    if HEADLESS:
        pygame.display.init()
        # Create a tiny dummy surface so font rendering works
        pygame.display.set_mode((1, 1))

    state = SimState()

    if HEADLESS:
        print(f"[HEADLESS]  mode={C.CONTROL_MODE}  seed={C.SEED}  "
              f"spawn_rate={C.SPAWN_RATE}  duration={C.DURATION}s  "
              f"log={C.LOG_FILE if not C.DISABLE_LOG else 'disabled'}")
        try:
            while True:
                state.update()
                # No sleep — run as fast as possible.
                # Signal threads use real time.sleep(1) so timing stays accurate.
        except KeyboardInterrupt:
            pass
        metrics.close()
        state._print_summary()
        return

    renderer = Renderer(state)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                metrics.close()
                state._print_summary()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_p:
                    state.paused = not state.paused
                elif k == pygame.K_h:
                    renderer.show_hud = not renderer.show_hud
                elif k == pygame.K_v:
                    C.SHOW_VECTORS = not C.SHOW_VECTORS
                elif k == pygame.K_r:
                    renderer.cam_tx = renderer.cam_ty = 0.0
                elif k in (pygame.K_EQUALS, pygame.K_PLUS):
                    C.SPAWN_INTERVAL = max(0.2, C.SPAWN_INTERVAL * 0.8)
                    print(f"[Spawn up]   interval={C.SPAWN_INTERVAL:.2f}s")
                elif k == pygame.K_MINUS:
                    C.SPAWN_INTERVAL = min(5.0, C.SPAWN_INTERVAL * 1.25)
                    print(f"[Spawn down] interval={C.SPAWN_INTERVAL:.2f}s")
                elif k == pygame.K_F1:
                    state._print_summary()
                elif k == pygame.K_1:
                    renderer.cam_tx, renderer.cam_ty = 0.0, 0.0
                elif k == pygame.K_2:
                    renderer.cam_tx = float(C.OFFSETS[1][0])
                    renderer.cam_ty = float(C.OFFSETS[1][1])
                elif k == pygame.K_3:
                    renderer.cam_tx = float(C.OFFSETS[2][0])
                    renderer.cam_ty = float(C.OFFSETS[2][1])
                elif k == pygame.K_4:
                    renderer.cam_tx = float(C.OFFSETS[3][0])
                    renderer.cam_ty = float(C.OFFSETS[3][1])

        keys = pygame.key.get_pressed()
        state.update()
        renderer.render(keys)


if __name__ == "__main__":
    main()