# simulation.py
# =============================================================================
# CSC 480 — Advanced Traffic Intersection Simulation
# =============================================================================

import random
import time
import sys
import os
import csv
import math
import argparse
import json
from datetime import datetime, timezone

import torch
from neural.model import load_model
from neural.utils import encode_features

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="CSC 480 Traffic Simulation")
parser.add_argument("--headless",   action="store_true")
parser.add_argument("--mode",       default="adaptive",
                    choices=["fixed", "adaptive", "greedy", "random", "rl", "neural"])
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--duration",   type=int,   default=0)
parser.add_argument("--spawn-rate", type=float, default=1.0)
parser.add_argument("--log",        default="metrics.csv")
parser.add_argument("--no-log",     action="store_true")
parser.add_argument("--legacy-log", action="store_true")
parser.add_argument("--dt",         type=float, default=1.0 / 60.0)
parser.add_argument("--fixed-green-seconds", type=int, default=3)
parser.add_argument("--shared-green-seconds", type=int, default=3)
parser.add_argument("--results-dir", default="results")
parser.add_argument("--experiment-id", default="")
parser.add_argument("--run-id",       default="")
parser.add_argument("--rl-model-path", default="models/rl_qtable.json")
parser.add_argument("--rl-train", action="store_true")
parser.add_argument("--rl-alpha", type=float, default=0.2)
parser.add_argument("--rl-gamma", type=float, default=0.95)
parser.add_argument("--rl-epsilon", type=float, default=0.20)
parser.add_argument("--rl-epsilon-min", type=float, default=0.02)
parser.add_argument("--rl-epsilon-decay", type=float, default=0.9995)
parser.add_argument("--rl-starvation-t", type=float, default=None)
parser.add_argument(
    "--rl-w-throughput",
    type=float,
    default=None,
    help="Reward weight for increasing cars crossed between decisions.",
)
parser.add_argument(
    "--rl-w-wait-delta",
    type=float,
    default=None,
    help="Reward weight for reducing queue-weighted wait mass between decisions.",
)
parser.add_argument(
    "--rl-switch-penalty",
    type=float,
    default=None,
    help="Penalty applied when RL switches to a different phase.",
)
# Legacy reward knobs retained only for backward CLI compatibility.
parser.add_argument("--rl-w-maxwait-delta", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--rl-w-queue-delta", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--rl-w-cur-queue", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--rl-w-cur-wait-mass", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--rl-w-cur-maxwait", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--rl-w-imbalance", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--rl-w-starved", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--neural-model-path", default="models/neural_hybrid.pt")
parser.add_argument("--collect-neural-data", action="store_true")
parser.add_argument("--neural-data-path", default="data/neural/greedy_data.csv")
parser.add_argument("--teacher-policy", default="greedy", choices=["greedy", "hybrid"])
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
    FIXED_GREEN_SECONDS = max(1, args.fixed_green_seconds)
    SHARED_GREEN_SECONDS = max(1, args.shared_green_seconds)
    DEFAULT_GREEN    = {
        0: FIXED_GREEN_SECONDS,
        1: FIXED_GREEN_SECONDS,
        2: FIXED_GREEN_SECONDS,
        3: FIXED_GREEN_SECONDS,
    }
    DEFAULT_RED      = 150
    DEFAULT_YELLOW   = 3
    MIN_GREEN        = 3
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
    FIXED_DT         = max(args.dt, 1e-3)

    # Metrics
    METRIC_INTERVAL  = 5.0
    LOG_FILE         = args.log
    DISABLE_LOG      = args.no_log
    LEGACY_LOG       = args.legacy_log
    RESULTS_DIR      = args.results_dir
    EXPERIMENT_ID    = args.experiment_id.strip()
    RUN_ID           = args.run_id.strip()
    RL_MODEL_PATH    = args.rl_model_path
    RL_TRAIN         = args.rl_train
    RL_ALPHA         = args.rl_alpha
    RL_GAMMA         = args.rl_gamma
    RL_EPSILON       = args.rl_epsilon
    RL_EPSILON_MIN   = args.rl_epsilon_min
    RL_EPSILON_DECAY = args.rl_epsilon_decay
    RL_STARVATION_T  = args.rl_starvation_t if args.rl_starvation_t is not None else 35.0
    # Simplified RL reward: throughput increase + wait reduction - switch penalty.
    RL_W_THROUGHPUT = (
        args.rl_w_throughput
        if args.rl_w_throughput is not None
        else (args.rl_w_queue_delta if args.rl_w_queue_delta is not None else 1.00)
    )
    RL_W_WAIT_DELTA = args.rl_w_wait_delta if args.rl_w_wait_delta is not None else 0.050
    RL_SWITCH_PENALTY = (
        args.rl_switch_penalty if args.rl_switch_penalty is not None else 0.20
    )
    NEURAL_MODEL_PATH = args.neural_model_path
    COLLECT_NEURAL_DATA = args.collect_neural_data
    NEURAL_DATA_PATH = args.neural_data_path
    TEACHER_POLICY  = args.teacher_policy

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
RNG = random.Random(C.SEED)

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
# RUN ARTIFACTS
# =============================================================================
class RunArtifacts:
    def __init__(self):
        exp = C.EXPERIMENT_ID or datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S")
        run = C.RUN_ID or f"{C.CONTROL_MODE}_seed{C.SEED}_spawn{C.SPAWN_RATE:.2f}".replace(".", "p")
        self.experiment_id = exp
        self.run_id = run
        self.experiment_dir = os.path.join(C.RESULTS_DIR, exp)
        self.run_dir = os.path.join(self.experiment_dir, run)
        os.makedirs(self.run_dir, exist_ok=True)

        self.timeseries_path = os.path.join(self.run_dir, "intersection_timeseries.csv")
        self.summary_path = os.path.join(self.run_dir, "summary.json")
        self.config_path = os.path.join(self.run_dir, "config.json")

    def write_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump({
                "mode": C.CONTROL_MODE,
                "seed": C.SEED,
                "spawn_rate": C.SPAWN_RATE,
                "duration_s": C.DURATION,
                "fixed_dt_s": C.FIXED_DT,
                "headless": HEADLESS,
                "fixed_green_seconds": C.FIXED_GREEN_SECONDS,
                "shared_green_seconds": C.SHARED_GREEN_SECONDS,
                "rl_model_path": C.RL_MODEL_PATH,
                "rl_train": C.RL_TRAIN,
                "rl_alpha": C.RL_ALPHA,
                "rl_gamma": C.RL_GAMMA,
                "rl_epsilon": C.RL_EPSILON,
                "rl_epsilon_min": C.RL_EPSILON_MIN,
                "rl_epsilon_decay": C.RL_EPSILON_DECAY,
                "rl_starvation_t": C.RL_STARVATION_T,
                "rl_reward_mode": "throughput_wait_switch",
                "rl_w_throughput": C.RL_W_THROUGHPUT,
                "rl_w_queue_delta": C.RL_W_THROUGHPUT,
                "rl_w_wait_delta": C.RL_W_WAIT_DELTA,
                "rl_switch_penalty": C.RL_SWITCH_PENALTY,
                "neural_model_path": C.NEURAL_MODEL_PATH,
                "collect_neural_data": C.COLLECT_NEURAL_DATA,
                "neural_data_path": C.NEURAL_DATA_PATH,
                "teacher_policy": C.TEACHER_POLICY,
                "grid_rows": C.GRID_ROWS,
                "grid_cols": C.GRID_COLS,
                "run_id": self.run_id,
                "experiment_id": self.experiment_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

    def write_summary(self, summary):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

# =============================================================================
# METRICS ENGINE
# =============================================================================
class MetricsEngine:
    def __init__(self, artifacts: RunArtifacts):
        self.artifacts          = artifacts
        self.start_time         = 0.0
        self.last_log_time      = 0.0
        self.total_crossed      = [0] * C.NO_INTERSECTIONS
        self.total_spawned      = [0] * C.NO_INTERSECTIONS
        self.external_spawned   = [0] * C.NO_INTERSECTIONS
        self.transferred_in     = [0] * C.NO_INTERSECTIONS
        self.exited_world       = [0] * C.NO_INTERSECTIONS
        self.red_violations     = [0] * C.NO_INTERSECTIONS
        self.wait_samples       = [[] for _ in range(C.NO_INTERSECTIONS)]
        self.wait_samples_by_dir = [
            {d: [] for d in C.DIRECTION_NUMS.values()}
            for _ in range(C.NO_INTERSECTIONS)
        ]
        self.queue_history      = [[] for _ in range(C.NO_INTERSECTIONS)]
        self.timeseries_rows    = []
        self._csv_file          = None
        self._csv_writer        = None
        self._legacy_csv_file   = None
        self._legacy_writer     = None
        if not C.DISABLE_LOG:
            self._init_csv()

    def _init_csv(self):
        write_header = not os.path.exists(self.artifacts.timeseries_path)
        self._csv_file = open(self.artifacts.timeseries_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if write_header:
            self._csv_writer.writerow([
                "elapsed_s", "mode", "seed", "spawn_rate", "iid",
                "avg_wait_s", "p95_wait_s", "max_wait_s",
                "avg_queue", "max_queue",
                "throughput_per_min", "fairness_index",
                "total_crossed", "total_spawned",
                "external_spawned", "transferred_in",
                "exited_world", "red_light_violations",
                "queue_right", "queue_down", "queue_left", "queue_up",
            ])
        if C.LEGACY_LOG:
            legacy_header = not os.path.exists(C.LOG_FILE)
            self._legacy_csv_file = open(C.LOG_FILE, "a", newline="")
            self._legacy_writer = csv.writer(self._legacy_csv_file)
            if legacy_header:
                self._legacy_writer.writerow([
                    "timestamp", "elapsed_s", "mode", "seed", "spawn_rate", "iid",
                    "avg_wait_s", "p95_wait_s", "max_wait_s",
                    "avg_queue", "max_queue", "throughput_per_min", "fairness_index",
                    "total_crossed", "total_spawned",
                ])

    def vehicle_spawned(self, vehicle, sim_time, source="external"):
        self.total_spawned[vehicle.iid] += 1
        if source == "external":
            self.external_spawned[vehicle.iid] += 1
        else:
            self.transferred_in[vehicle.iid] += 1

    def vehicle_crossed(self, vehicle, sim_time, violated_red=False):
        iid = vehicle.iid
        self.total_crossed[iid] += 1
        if violated_red:
            self.red_violations[iid] += 1
        waited = vehicle.total_wait_s
        if vehicle.waiting and vehicle.wait_start is not None:
            waited += max(0.0, sim_time - vehicle.wait_start)
        self.wait_samples[iid].append(waited)
        self.wait_samples_by_dir[iid][vehicle.direction].append(waited)

    def vehicle_exited(self, iid):
        self.exited_world[iid] += 1

    def record_queue(self, iid, q):
        self.queue_history[iid].append(q)

    def _all_waits(self, iid, vehicles_state, sim_time):
        waits = list(self.wait_samples[iid])
        for d in C.DIRECTION_NUMS.values():
            for lane in range(3):
                for v in vehicles_state[iid][d][lane]:
                    if v.crossed != 0:
                        continue
                    w = v.total_wait_s
                    if v.waiting and v.wait_start is not None:
                        w += max(0.0, sim_time - v.wait_start)
                    waits.append(w)
        return waits

    def _all_waits_dir(self, iid, direction, vehicles_state, sim_time):
        waits = list(self.wait_samples_by_dir[iid][direction])
        for lane in range(3):
            for v in vehicles_state[iid][direction][lane]:
                if v.crossed != 0:
                    continue
                w = v.total_wait_s
                if v.waiting and v.wait_start is not None:
                    w += max(0.0, sim_time - v.wait_start)
                waits.append(w)
        return waits

    def get_avg_wait(self, iid, vehicles_state, sim_time):
        s = self._all_waits(iid, vehicles_state, sim_time)
        return sum(s) / len(s) if s else 0.0

    def get_p95_wait(self, iid, vehicles_state, sim_time):
        s = sorted(self._all_waits(iid, vehicles_state, sim_time))
        if not s:
            return 0.0
        return s[min(int(0.95 * len(s)), len(s) - 1)]

    def get_max_wait(self, iid, vehicles_state, sim_time):
        return max(self._all_waits(iid, vehicles_state, sim_time), default=0.0)

    def get_avg_wait_dir(self, iid, direction, vehicles_state, sim_time):
        s = self._all_waits_dir(iid, direction, vehicles_state, sim_time)
        return sum(s) / len(s) if s else 0.0

    def get_avg_queue(self, iid):
        h = self.queue_history[iid]
        return sum(h) / len(h) if h else 0.0

    def get_max_queue(self, iid):
        return max(self.queue_history[iid], default=0)

    def get_throughput(self, iid, sim_time):
        elapsed = max(sim_time - self.start_time, 1.0)
        return self.total_crossed[iid] / elapsed * 60.0

    def get_fairness_index(self, iid, vehicles_state):
        qs = [
            sum(1 for lane in range(3)
                for v in vehicles_state[iid][d][lane]
                if v.crossed == 0)
            for d in C.DIRECTION_NUMS.values()
        ]
        total = sum(qs)
        if total == 0:
            return 1.0
        n   = len(qs)
        num = total ** 2
        den = n * sum(q * q for q in qs)
        return num / den if den > 0 else 1.0

    def maybe_log(self, sim_time, vehicles_state):
        if C.DISABLE_LOG or self._csv_writer is None:
            return
        if sim_time - self.last_log_time < C.METRIC_INTERVAL:
            return
        self.last_log_time = sim_time
        elapsed = sim_time - self.start_time
        ts = datetime.now(timezone.utc).isoformat()
        for iid in range(C.NO_INTERSECTIONS):
            qs = [
                sum(1 for lane in range(3)
                    for v in vehicles_state[iid][d][lane]
                    if v.crossed == 0)
                for d in C.DIRECTION_NUMS.values()
            ]
            row = [
                f"{elapsed:.1f}", C.CONTROL_MODE, C.SEED, C.SPAWN_RATE, iid,
                f"{self.get_avg_wait(iid, vehicles_state, sim_time):.3f}",
                f"{self.get_p95_wait(iid, vehicles_state, sim_time):.3f}",
                f"{self.get_max_wait(iid, vehicles_state, sim_time):.3f}",
                f"{self.get_avg_queue(iid):.3f}",
                f"{self.get_max_queue(iid)}",
                f"{self.get_throughput(iid, sim_time):.3f}",
                f"{self.get_fairness_index(iid, vehicles_state):.4f}",
                self.total_crossed[iid],
                self.total_spawned[iid],
                self.external_spawned[iid],
                self.transferred_in[iid],
                self.exited_world[iid],
                self.red_violations[iid],
                qs[0], qs[1], qs[2], qs[3],
            ]
            self._csv_writer.writerow(row)
            if self._legacy_writer is not None:
                self._legacy_writer.writerow([
                    ts, f"{elapsed:.1f}", C.CONTROL_MODE, C.SEED, C.SPAWN_RATE, iid,
                    f"{self.get_avg_wait(iid, vehicles_state, sim_time):.3f}",
                    f"{self.get_p95_wait(iid, vehicles_state, sim_time):.3f}",
                    f"{self.get_max_wait(iid, vehicles_state, sim_time):.3f}",
                    f"{self.get_avg_queue(iid):.3f}",
                    f"{self.get_max_queue(iid)}",
                    f"{self.get_throughput(iid, sim_time):.3f}",
                    f"{self.get_fairness_index(iid, vehicles_state):.4f}",
                    self.total_crossed[iid],
                    self.total_spawned[iid],
                ])
        self._csv_file.flush()
        if self._legacy_csv_file:
            self._legacy_csv_file.flush()

    def snapshot(self, iid, vehicles_state, sim_time):
        qs = [
            sum(1 for lane in range(3)
                for v in vehicles_state[iid][d][lane]
                if v.crossed == 0)
            for d in C.DIRECTION_NUMS.values()
        ]
        return {
            "avg_wait":   self.get_avg_wait(iid, vehicles_state, sim_time),
            "avg_wait_right": self.get_avg_wait_dir(iid, "right", vehicles_state, sim_time),
            "avg_wait_down":  self.get_avg_wait_dir(iid, "down", vehicles_state, sim_time),
            "avg_wait_left":  self.get_avg_wait_dir(iid, "left", vehicles_state, sim_time),
            "avg_wait_up":    self.get_avg_wait_dir(iid, "up", vehicles_state, sim_time),
            "p95_wait":   self.get_p95_wait(iid, vehicles_state, sim_time),
            "max_wait":   self.get_max_wait(iid, vehicles_state, sim_time),
            "throughput": self.get_throughput(iid, sim_time),
            "avg_queue":  self.get_avg_queue(iid),
            "max_queue":  self.get_max_queue(iid),
            "queues":     qs,
            "crossed":    self.total_crossed[iid],
            "spawned":    self.total_spawned[iid],
            "external_spawned": self.external_spawned[iid],
            "transferred_in": self.transferred_in[iid],
            "exited_world": self.exited_world[iid],
            "red_light_violations": self.red_violations[iid],
            "fairness":   self.get_fairness_index(iid, vehicles_state),
        }

    def close(self):
        if self._csv_file:
            self._csv_file.close()
        if self._legacy_csv_file:
            self._legacy_csv_file.close()

metrics = None

# =============================================================================
# TRAFFIC SIGNAL
# =============================================================================
class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red        = red
        self.yellow     = yellow
        self.green      = green
        self.signalText = ""


class TabularRLPolicy:
    def __init__(self, path, training):
        self.path = path
        self.training = training
        self.alpha = C.RL_ALPHA
        self.gamma = C.RL_GAMMA
        self.epsilon = C.RL_EPSILON
        self.epsilon_min = C.RL_EPSILON_MIN
        self.epsilon_decay = C.RL_EPSILON_DECAY
        self.q = {}
        self.updates = 0
        if os.path.exists(self.path):
            self.load()

    def _key(self, state):
        return "|".join(str(x) for x in state)

    def _ensure(self, key, prior=None):
        if key not in self.q:
            if prior is None:
                self.q[key] = [0.0, 0.0, 0.0, 0.0]
            else:
                self.q[key] = [float(v) for v in prior]
        return self.q[key]

    def act(self, state, prior=None):
        key = self._key(state)
        values = self._ensure(key, prior=prior)
        if self.training and RNG.random() < self.epsilon:
            return RNG.randint(0, 3)
        best = max(values)
        best_idxs = [i for i, v in enumerate(values) if v == best]
        return RNG.choice(best_idxs)

    def update(self, state, action, reward, next_state):
        if not self.training:
            return
        key = self._key(state)
        next_key = self._key(next_state)
        q_s = self._ensure(key)
        q_n = self._ensure(next_key)
        td_target = reward + self.gamma * max(q_n)
        q_s[action] += self.alpha * (td_target - q_s[action])
        self.updates += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.q = {k: list(v) for k, v in data.get("q_table", {}).items()}
        if "epsilon" in data:
            self.epsilon = float(data["epsilon"])

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({
                "q_table": self.q,
                "epsilon": self.epsilon,
                "updates": self.updates,
                "seed": C.SEED,
                "saved_utc": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)


RL_POLICY = None
NEURAL_POLICY = None
NEURAL_DATA_COLLECTOR = None


class NeuralDataCollector:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._header_written = os.path.exists(path) and os.path.getsize(path) > 0

    def append(self, features, action):
        write_header = not self._header_written
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "phase0", "phase1", "phase2", "phase3",
                    "q_right", "q_down", "q_left", "q_up",
                    "avg_wait_right", "avg_wait_down", "avg_wait_left", "avg_wait_up",
                    "max_wait_right", "max_wait_down", "max_wait_left", "max_wait_up",
                    "downstream_right", "downstream_down", "downstream_left", "downstream_up",
                    "action",
                ])
                self._header_written = True
            w.writerow(list(features) + [int(action)])

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

    def get_queue_lengths(self, vehicles_state):
        iid = self.iid
        return [
            sum(1 for lane in range(3)
                for v in vehicles_state[iid][C.DIRECTION_NUMS[dn]][lane]
                if v.crossed == 0)
            for dn in range(4)
        ]

    def get_wait_stats(self, vehicles_state, now_s):
        iid = self.iid
        avg_waits = []
        max_waits = []
        for dn in range(4):
            direction = C.DIRECTION_NUMS[dn]
            waits = []
            for lane in range(3):
                for v in vehicles_state[iid][direction][lane]:
                    if v.crossed != 0:
                        continue
                    w = v.total_wait_s
                    if v.waiting and v.wait_start is not None:
                        w += max(0.0, now_s - v.wait_start)
                    waits.append(w)
            avg_waits.append(sum(waits) / len(waits) if waits else 0.0)
            max_waits.append(max(waits) if waits else 0.0)
        return avg_waits, max_waits
    
    def get_downstream_totals(self, vehicles_state):
        vals = []
        row = self.iid // C.GRID_COLS
        col = self.iid % C.GRID_COLS

        for dn in range(4):
            nr, nc = row, col
            if dn == 0:
                nc += 1
            elif dn == 1:
                nr += 1
            elif dn == 2:
                nc -= 1
            else:
                nr -= 1

            if nr < 0 or nr >= C.GRID_ROWS or nc < 0 or nc >= C.GRID_COLS:
                vals.append(0)
                continue

            nid = nr * C.GRID_COLS + nc
            total = 0
            for d in C.DIRECTION_NUMS.values():
                for lane in range(3):
                    total += sum(
                        1 for v in vehicles_state[nid][d][lane]
                        if v.crossed == 0
                    )
            vals.append(total)

        return vals

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

    def next_green_duration(self, signals_list, cur_green, vehicles_state, now_s):
        return C.DEFAULT_GREEN[cur_green[self.iid]]

    def choose_next_phase(self, vehicles_state, cur_green, now_s):
        # Default: round-robin
        return (cur_green[self.iid] + 1) % C.NO_OF_SIGNALS


class FixedTimeController(SignalController):
    """
    PAIR 1 — Baseline.
    Fixed cycle, fixed duration. Every direction gets exactly the same
    green time in strict round-robin order regardless of traffic load.
    """
    def next_green_duration(self, signals_list, cur_green, vehicles_state, now_s):
        return C.DEFAULT_GREEN[cur_green[self.iid]]

    def choose_next_phase(self, vehicles_state, cur_green, now_s):
        return (cur_green[self.iid] + 1) % C.NO_OF_SIGNALS


class GreedyController(SignalController):
    """
    PAIR 2 — Greedy / Rule-Based.
    Always grants green to the direction with the most waiting vehicles.
    Duration scales with queue depth.
    AI class: goal-based / simple-reflex agent.
    """
    def choose_next_phase(self, vehicles_state, cur_green, now_s):
        iid    = self.iid
        queues = [
            sum(1 for lane in range(3)
                for v in vehicles_state[iid][C.DIRECTION_NUMS[dn]][lane]
                if v.crossed == 0)
            for dn in range(4)
        ]
        return max(range(4), key=lambda i: queues[i])

    def next_green_duration(self, signals_list, cur_green, vehicles_state, now_s):
        iid = self.iid
        d   = C.DIRECTION_NUMS[cur_green[iid]]
        q   = sum(1 for lane in range(3)
                  for v in vehicles_state[iid][d][lane]
                  if v.crossed == 0)
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
        self._last_green_time = {dn: 0.0 for dn in range(4)}

    def _utility(self, vehicles_state, cur_green, now_s):
        iid  = self.iid
        util = {}
        for dn in range(4):
            d = C.DIRECTION_NUMS[dn]
            q = sum(1 for lane in range(3)
                    for v in vehicles_state[iid][d][lane]
                    if v.crossed == 0)
            waits = [
                now_s - v.wait_start
                for lane in range(3)
                for v in vehicles_state[iid][d][lane]
                if v.crossed == 0 and v.waiting and v.wait_start is not None
            ]
            avg_wait   = sum(waits) / len(waits) if waits else 0.0
            starvation = max(
                0.0,
                (now_s - self._last_green_time.get(dn, now_s))
                - C.ADAPTIVE_STARVATION_T
            )
            util[dn] = (
                C.ADAPTIVE_W_QUEUE      * q        +
                C.ADAPTIVE_W_WAIT       * avg_wait +
                C.ADAPTIVE_W_STARVATION * starvation
            )
        return util

    def choose_next_phase(self, vehicles_state, cur_green, now_s):
        self._last_green_time[cur_green[self.iid]] = now_s
        util = self._utility(vehicles_state, cur_green, now_s)
        return max(util, key=util.get)

    def next_green_duration(self, signals_list, cur_green, vehicles_state, now_s):
        iid = self.iid
        d   = C.DIRECTION_NUMS[cur_green[iid]]
        q   = sum(1 for lane in range(3)
                  for v in vehicles_state[iid][d][lane]
                  if v.crossed == 0)
        return int(max(C.MIN_GREEN, min(C.MAX_GREEN, C.MIN_GREEN + q * 1.8)))


class RandomController(SignalController):
    """Null baseline — random durations AND random phase order."""
    def next_green_duration(self, signals_list, cur_green, vehicles_state, now_s):
        return RNG.randint(C.MIN_GREEN, C.MAX_GREEN)

    def choose_next_phase(self, vehicles_state, cur_green, now_s):
        return RNG.randint(0, 3)


class RLController(SignalController):
    """
    Pressure- and delay-aware Q-learning controller.

    Improvements over the original version:
      - richer tabular state with queue, wait, pressure, and starvation summary
      - heuristic-prior Q initialisation for unseen states
      - simplified shaped reward based on throughput increase, wait reduction,
        and switch cost
      - explicit option to keep the current phase by selecting the same phase
        again when green expires (handled in SimState._step_signals_one_second)
    """
    def __init__(self, iid):
        super().__init__(iid)
        self.prev_state = None
        self.prev_action = None
        self.prev_metrics = None
        self.prev_switched = False
        self.last_served = {dn: 0.0 for dn in range(4)}

    def _downstream_iid(self, dn):
        row = self.iid // C.GRID_COLS
        col = self.iid % C.GRID_COLS
        if dn == 0:
            col += 1
        elif dn == 1:
            row += 1
        elif dn == 2:
            col -= 1
        else:
            row -= 1
        if row < 0 or row >= C.GRID_ROWS or col < 0 or col >= C.GRID_COLS:
            return None
        return row * C.GRID_COLS + col

    def _downstream_totals(self, vehicles_state):
        vals = []
        for dn in range(4):
            nid = self._downstream_iid(dn)
            if nid is None:
                vals.append(0)
                continue
            total = 0
            for d in C.DIRECTION_NUMS.values():
                for lane in range(3):
                    total += sum(1 for v in vehicles_state[nid][d][lane] if v.crossed == 0)
            vals.append(total)
        return vals

    def _collect_metrics(self, vehicles_state, cur_green, now_s):
        queues = self.get_queue_lengths(vehicles_state)
        avg_waits, max_waits = self.get_wait_stats(vehicles_state, now_s)
        downstream = self._downstream_totals(vehicles_state)
        pressures = [queues[i] - 0.35 * downstream[i] + 0.12 * avg_waits[i] for i in range(4)]
        since_served = [now_s - self.last_served.get(i, 0.0) for i in range(4)]
        total_q = sum(queues)
        wait_mass = sum(q * w for q, w in zip(queues, avg_waits))
        max_wait = max(max_waits) if max_waits else 0.0
        imbalance = max(queues) - min(queues) if queues else 0
        starved_count = sum(1 for t in since_served if t > C.RL_STARVATION_T)
        dom_q = max(range(4), key=lambda i: (queues[i], avg_waits[i], -i))
        dom_wait = max(range(4), key=lambda i: (avg_waits[i], queues[i], -i))
        dom_pressure = max(range(4), key=lambda i: (pressures[i], avg_waits[i], -i))
        starved_dir = 4 if max(since_served) <= C.RL_STARVATION_T else max(range(4), key=lambda i: since_served[i])
        return {
            "queues": queues,
            "avg_waits": avg_waits,
            "max_waits": max_waits,
            "downstream": downstream,
            "pressures": pressures,
            "since_served": since_served,
            "total_q": total_q,
            "wait_mass": wait_mass,
            "max_wait": max_wait,
            "imbalance": imbalance,
            "crossed": metrics.total_crossed[self.iid] if metrics is not None else 0,
            "dom_q": dom_q,
            "dom_wait": dom_wait,
            "dom_pressure": dom_pressure,
            "starved_dir": starved_dir,
            "starved_count": starved_count,
            "cur_phase": cur_green[self.iid],
        }

    def _state(self, m):
        q_bucket = min(m["total_q"] // 4, 7)
        max_wait_bucket = min(int(m["max_wait"] // 10), 7)
        imbalance_bucket = min(int(m["imbalance"] // 2), 5)
        cur_phase = m["cur_phase"]
        cur_q_bucket = min(m["queues"][cur_phase] // 3, 5)
        cur_wait_bucket = min(int(m["avg_waits"][cur_phase] // 8), 7)
        return (
            cur_phase,
            m["dom_q"],
            m["dom_wait"],
            m["dom_pressure"],
            q_bucket,
            max_wait_bucket,
            imbalance_bucket,
            m["starved_dir"],
            cur_q_bucket,
            cur_wait_bucket,
        )

    def _prior_values(self, m):
        vals = []
        for a in range(4):
            starvation_bonus = 1.5 if m["since_served"][a] > C.RL_STARVATION_T else 0.0
            score = (
                1.10 * m["queues"][a]
                + 0.12 * m["avg_waits"][a]
                + 0.05 * m["max_waits"][a]
                + 0.85 * max(m["pressures"][a], 0.0)
                + starvation_bonus
            )
            vals.append(score / 8.0)
        return vals

    def _reward(self, prev_m, cur_m):
        delta_throughput = cur_m["crossed"] - prev_m["crossed"]
        delta_wait = prev_m["wait_mass"] - cur_m["wait_mass"]
        reward = C.RL_W_THROUGHPUT * delta_throughput + C.RL_W_WAIT_DELTA * delta_wait
        if self.prev_switched:
            reward -= C.RL_SWITCH_PENALTY
        return reward

    def choose_next_phase(self, vehicles_state, cur_green, now_s):
        global RL_POLICY
        if RL_POLICY is None:
            return RNG.randint(0, 3)

        metrics_now = self._collect_metrics(vehicles_state, cur_green, now_s)
        state = self._state(metrics_now)
        if self.prev_state is not None and self.prev_action is not None and self.prev_metrics is not None:
            reward = self._reward(self.prev_metrics, metrics_now)
            RL_POLICY.update(self.prev_state, self.prev_action, reward, state)

        action = RL_POLICY.act(state, prior=self._prior_values(metrics_now))
        self.prev_state = state
        self.prev_action = action
        self.prev_metrics = metrics_now
        self.prev_switched = (action != cur_green[self.iid])
        self.last_served[cur_green[self.iid]] = now_s
        return action

    def next_green_duration(self, signals_list, cur_green, vehicles_state, now_s):
        queues = self.get_queue_lengths(vehicles_state)
        avg_waits, max_waits = self.get_wait_stats(vehicles_state, now_s)
        cur = cur_green[self.iid]
        q = queues[cur]
        avg_w = avg_waits[cur]
        max_w = max_waits[cur]
        dur = C.MIN_GREEN + 1.25 * q + 0.10 * avg_w + 0.04 * max_w
        return int(max(C.MIN_GREEN, min(C.MAX_GREEN, dur)))

class HybridTeacherController(SignalController):

    def __init__(self, iid):
        super().__init__(iid)
        self._last_green_time = {dn: 0.0 for dn in range(4)}

    def _norm(self, x, cap):
        return min(float(x), float(cap)) / float(cap)

    def _score(self, dn, cur_phase, queues, avg_waits, max_waits, downstream, now_s):
        since_served = now_s - self._last_green_time.get(dn, now_s)
        starvation = max(0.0, since_served - 22.0)

        # Soft downstream-aware pressure
        pressure_raw = max(0.0, queues[dn] - 0.18 * downstream[dn])

        q_n         = self._norm(queues[dn], 20)
        avg_w_n     = self._norm(avg_waits[dn], 40)
        max_w_n     = self._norm(max_waits[dn], 80)
        down_n      = self._norm(downstream[dn], 20)
        pressure_n  = self._norm(pressure_raw, 20)
        starve_n    = self._norm(starvation, 40)

        # Small hold bias for current phase if it is still competitive.
        keep_bias = 0.0
        if dn == cur_phase:
            keep_bias = 0.10 + 0.05 * q_n

        return (
            1.30 * q_n
            + 0.50 * avg_w_n
            + 0.75 * max_w_n
            + 0.60 * pressure_n
            + 0.55 * starve_n
            - 0.10 * down_n
            + keep_bias
        )

    def choose_next_phase(self, vehicles_state, cur_green, now_s):
        cur_phase = cur_green[self.iid]
        self._last_green_time[cur_phase] = now_s

        queues = self.get_queue_lengths(vehicles_state)
        avg_waits, max_waits = self.get_wait_stats(vehicles_state, now_s)
        downstream = self.get_downstream_totals(vehicles_state)

        scores = [
            self._score(dn, cur_phase, queues, avg_waits, max_waits, downstream, now_s)
            for dn in range(4)
        ]

        # Deterministic tie-breaks reduce label noise for supervised learning.
        best_dn = max(
            range(4),
            key=lambda dn: (
                scores[dn],
                max_waits[dn],
                avg_waits[dn],
                queues[dn],
                -abs(dn - cur_phase),   # slightly prefer staying close to current if tied
                -dn
            )
        )
        return best_dn

    def next_green_duration(self, signals_list, cur_green, vehicles_state, now_s):
        cur = cur_green[self.iid]
        queues = self.get_queue_lengths(vehicles_state)
        avg_waits, max_waits = self.get_wait_stats(vehicles_state, now_s)
        downstream = self.get_downstream_totals(vehicles_state)

        q = queues[cur]
        avg_w = avg_waits[cur]
        max_w = max_waits[cur]
        down = downstream[cur]

        dur = (
            C.MIN_GREEN
            + 1.10 * q
            + 0.08 * avg_w
            + 0.03 * max_w
            - 0.05 * down
        )
        return int(max(C.MIN_GREEN, min(C.MAX_GREEN, dur)))


class NeuralController(SignalController):
    def __init__(self, iid):
        super().__init__(iid)

    def choose_next_phase(self, vehicles_state, cur_green, now_s):
        global NEURAL_POLICY
        if NEURAL_POLICY is None:
            return RNG.randint(0, 3)

        queues = self.get_queue_lengths(vehicles_state)
        avg_waits, max_waits = self.get_wait_stats(vehicles_state, now_s)
        downstream = self.get_downstream_totals(vehicles_state)
        feats = encode_features(
            cur_green[self.iid], queues, avg_waits, max_waits, downstream
        )
        x = torch.tensor([feats], dtype=torch.float32)
        with torch.no_grad():
            logits = NEURAL_POLICY(x)
            action = int(torch.argmax(logits, dim=1).item())
        return action

    def next_green_duration(self, signals_list, cur_green, vehicles_state, now_s):
        cur = cur_green[self.iid]
        queues = self.get_queue_lengths(vehicles_state)
        avg_waits, max_waits = self.get_wait_stats(vehicles_state, now_s)
        downstream = self.get_downstream_totals(vehicles_state)

        q = queues[cur]
        avg_w = avg_waits[cur]
        max_w = max_waits[cur]
        down = downstream[cur]

        dur = (
            C.MIN_GREEN
            + 1.10 * q
            + 0.08 * avg_w
            + 0.03 * max_w
            - 0.05 * down
        )
        return int(max(C.MIN_GREEN, min(C.MAX_GREEN, dur)))


CONTROLLER_MAP = {
    "fixed":    FixedTimeController,
    "adaptive": AdaptiveController,
    "greedy":   GreedyController,
    "random":   RandomController,
    "rl":       RLController,
    "neural":   NeuralController,
}

# =============================================================================
# PARTICLES
# =============================================================================
class Particle:
    __slots__ = ["x", "y", "vx", "vy", "life", "max_life", "color", "size"]
    _OX = {'right': -1, 'left': 1, 'down': 0, 'up': 0}
    _OY = {'right': 0,  'left': 0, 'down': -1, 'up': 1}

    def __init__(self, x, y, direction):
        self.x        = x + RNG.uniform(-3, 3)
        self.y        = y + RNG.uniform(-3, 3)
        self.vx       = self._OX[direction] * RNG.uniform(0.4, 1.4) + RNG.uniform(-0.3, 0.3)
        self.vy       = self._OY[direction] * RNG.uniform(0.4, 1.4) + RNG.uniform(-0.3, 0.3)
        self.max_life = RNG.randint(18, 42)
        self.life     = self.max_life
        g             = RNG.randint(130, 195)
        self.color    = (g, g, g)
        self.size     = RNG.uniform(1.5, 3.5)

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
                 sim_group, sim_time):
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
        self._in_lane         = False

        self.x = _spawn_x[iid][direction][lane]
        self.y = _spawn_y[iid][direction][lane]

        # ---- Image loading — headless-safe ----
        self.image = load_image(f"images/{direction}/{vehicleClass}.png")
        self.rect  = self.image.get_rect()

        # Register
        self._attach_to_lane(iid)

        # ---- Advance spawn stack ----
        if direction == 'right':
            _spawn_x[iid][direction][lane] -= self.rect.width  + C.STOPPING_GAP
        elif direction == 'left':
            _spawn_x[iid][direction][lane] += self.rect.width  + C.STOPPING_GAP
        elif direction == 'down':
            _spawn_y[iid][direction][lane] -= self.rect.height + C.STOPPING_GAP
        elif direction == 'up':
            _spawn_y[iid][direction][lane] += self.rect.height + C.STOPPING_GAP

        metrics.vehicle_spawned(self, sim_time, source="external")
        sim_group.add(self)

    def _lane_bucket(self, iid=None):
        target_iid = self.iid if iid is None else iid
        return _vehicles[target_iid][self.direction][self.lane]

    def _set_stop_position(self, iid):
        lane_bucket = self._lane_bucket(iid)
        if self.index > 0 and lane_bucket[self.index - 1].crossed == 0:
            prev = lane_bucket[self.index - 1]
            if self.direction == 'right':
                self.stop = prev.stop - prev.rect.width - C.STOPPING_GAP
            elif self.direction == 'left':
                self.stop = prev.stop + prev.rect.width + C.STOPPING_GAP
            elif self.direction == 'down':
                self.stop = prev.stop - prev.rect.height - C.STOPPING_GAP
            elif self.direction == 'up':
                self.stop = prev.stop + prev.rect.height + C.STOPPING_GAP
        else:
            self.stop = default_stop_pos(iid, self.direction)

    def _attach_to_lane(self, iid):
        self.iid = iid
        lane_bucket = self._lane_bucket()
        lane_bucket.append(self)
        self.index = len(lane_bucket) - 1
        self._in_lane = True
        self._set_stop_position(iid)

    def _detach_from_lane(self):
        if not self._in_lane:
            return
        lane_bucket = self._lane_bucket()
        if 0 <= self.index < len(lane_bucket) and lane_bucket[self.index] is self:
            remove_at = self.index
        else:
            try:
                remove_at = lane_bucket.index(self)
            except ValueError:
                self._in_lane = False
                self.index = -1
                return
        lane_bucket.pop(remove_at)
        for idx in range(remove_at, len(lane_bucket)):
            lane_bucket[idx].index = idx
        self._in_lane = False
        self.index = -1

    def _handoff_to(self, new_iid, sim_time):
        self._detach_from_lane()
        self.crossed = 0
        self.waiting = False
        self.wait_start = None
        self._attach_to_lane(new_iid)
        metrics.vehicle_spawned(self, sim_time, source="transfer")

    def _edge_handoff_target(self, w, h):
        row = self.iid // C.GRID_COLS
        col = self.iid % C.GRID_COLS
        ox, oy = C.OFFSETS[self.iid]
        left = ox
        right = ox + C.TILE_W
        top = oy
        bottom = oy + C.TILE_H

        if self.direction == 'right' and self.x + w >= right:
            col += 1
        elif self.direction == 'left' and self.x <= left:
            col -= 1
        elif self.direction == 'down' and self.y + h >= bottom:
            row += 1
        elif self.direction == 'up' and self.y <= top:
            row -= 1
        else:
            return None

        if row < 0 or row >= C.GRID_ROWS or col < 0 or col >= C.GRID_COLS:
            return -1
        return row * C.GRID_COLS + col

    def despawn(self):
        metrics.vehicle_exited(self.iid)
        self._detach_from_lane()
        self.kill()

    # ------------------------------------------------------------------
    def move(self, cur_green, cur_yellow, sim_time, dt_scale):
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
        my_green = (cg == my_signal) and (cy == 0)
        # Count red violations only when another direction owns green.
        # Yellow on the same phase is treated as legal "clearance" time.
        my_red = (cg != my_signal)

        # ---- 1. Crossing detection ----
        sl = stop_line(iid, d)
        if self.crossed == 0:
            hit = (
                (d == 'right' and self.x + w >= sl) or
                (d == 'left'  and self.x       <= sl) or
                (d == 'down'  and self.y + h   >= sl) or
                (d == 'up'    and self.y        <= sl)
            )
            if hit:
                self.crossed = 1
                metrics.vehicle_crossed(self, sim_time, violated_red=my_red)

        # ---- 2. Is it my green? ----
        # Green only when:  the current green phase matches MY signal slot
        #                   AND we are not in yellow

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
            self.current_speed = min(self.current_speed + C.ACCEL * dt_scale, target)
        else:
            self.current_speed = max(self.current_speed - C.DECEL * dt_scale, target)

        # ---- 6. Wait tracking ----
        if self.current_speed < 0.05:
            if not self.waiting:
                self.waiting    = True
                self.wait_start = sim_time
        else:
            if self.waiting:
                self.waiting = False
                if self.wait_start is not None:
                    self.total_wait_s += sim_time - self.wait_start
                self.wait_start = None

        # ---- 7. Apply movement ----
        # Prevent momentum overshoot through the stop point when signal is not
        # green for this direction. This keeps red/yellow compliance strict.
        must_hold_at_stop = (self.crossed == 0 and not my_green)
        if d == 'right':
            proposed = self.x + self.current_speed * dt_scale
            if must_hold_at_stop:
                proposed = min(proposed, self.stop - w)
                if proposed >= self.stop - w:
                    self.current_speed = 0.0
            self.x = proposed
        elif d == 'left':
            proposed = self.x - self.current_speed * dt_scale
            if must_hold_at_stop:
                proposed = max(proposed, self.stop)
                if proposed <= self.stop:
                    self.current_speed = 0.0
            self.x = proposed
        elif d == 'down':
            proposed = self.y + self.current_speed * dt_scale
            if must_hold_at_stop:
                proposed = min(proposed, self.stop - h)
                if proposed >= self.stop - h:
                    self.current_speed = 0.0
            self.y = proposed
        elif d == 'up':
            proposed = self.y - self.current_speed * dt_scale
            if must_hold_at_stop:
                proposed = max(proposed, self.stop)
                if proposed <= self.stop:
                    self.current_speed = 0.0
            self.y = proposed

        # ---- 8. Particles ----
        if C.PARTICLE_EFFECTS and self.current_speed > 0.5:
            self._particle_timer += 1
            if self._particle_timer % 8 == 0:
                _particle_pool.append(Particle(self.x, self.y, d))

        # ---- 9. Tile boundary transition ----
        # Move to neighboring tile and continue under that intersection's
        # signal rules. Despawn only when leaving the outer world boundary.
        target_iid = self._edge_handoff_target(w, h)
        if target_iid == -1:
            self._offscreen = True
        elif target_iid is not None and target_iid != self.iid:
            self._handoff_to(target_iid, sim_time)

# =============================================================================
# SIMULATION STATE
# =============================================================================
class SimState:
    def __init__(self):
        global _spawn_x, _spawn_y, _vehicles

        self.sim_group       = pygame.sprite.Group()
        self.signals         = [[] for _ in range(C.NO_INTERSECTIONS)]
        self.cur_green       = [0] * C.NO_INTERSECTIONS
        self.next_green      = [1] * C.NO_INTERSECTIONS
        self.cur_yellow      = [0] * C.NO_INTERSECTIONS
        self.signal_accum_s  = [0.0] * C.NO_INTERSECTIONS
        self.spawn_accum_s   = 0.0
        self.queue_accum_s   = 0.0

        ctrl_cls = CONTROLLER_MAP.get(C.CONTROL_MODE, AdaptiveController)
        self.controllers = [ctrl_cls(iid) for iid in range(C.NO_INTERSECTIONS)]

        self.teacher_controllers = None
        if C.COLLECT_NEURAL_DATA and C.TEACHER_POLICY == "hybrid":
            self.teacher_controllers = [
                HybridTeacherController(iid) for iid in range(C.NO_INTERSECTIONS)
            ]

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

        self.vehicles      = _vehicles
        self.start_time    = 0.0
        self.sim_time      = 0.0
        self.paused        = False
        self.frame_count   = 0

        for iid in range(C.NO_INTERSECTIONS):
            self._init_signals(iid)
            self._activate_green(iid)

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
    
    def _effective_controller(self, iid):
        if self.teacher_controllers is not None:
            return self.teacher_controllers[iid]
        return self.controllers[iid]
    
    def _compute_green_duration(self, iid, ctrl):
        if C.CONTROL_MODE != "fixed":
            return int(max(C.MIN_GREEN, min(C.MAX_GREEN, C.SHARED_GREEN_SECONDS)))
        return int(max(C.MIN_GREEN, min(
            C.MAX_GREEN,
            ctrl.next_green_duration(
                self.signals, self.cur_green, self.vehicles, self.sim_time
            )
        )))

    def _activate_green(self, iid):
        ctrl = self._effective_controller(iid)
        dur = self._compute_green_duration(iid, ctrl)
        self.signals[iid][self.cur_green[iid]].green = dur
        self.signals[iid][self.cur_green[iid]].yellow = C.DEFAULT_YELLOW
        ng = self.next_green[iid]
        self.signals[iid][ng].red = (
            self.signals[iid][self.cur_green[iid]].yellow + dur
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

    def _step_signals_one_second(self, iid):
        self._tick(iid)
        cg = self.cur_green[iid]
        if self.cur_yellow[iid]:
            if self.signals[iid][cg].yellow == 0:
                self.cur_yellow[iid] = 0
                self.signals[iid][cg].green  = C.DEFAULT_GREEN[cg]
                self.signals[iid][cg].yellow = C.DEFAULT_YELLOW
                self.signals[iid][cg].red    = C.DEFAULT_RED
                self.cur_green[iid] = self.next_green[iid]
                self.next_green[iid] = (self.cur_green[iid] + 1) % C.NO_OF_SIGNALS
                self._activate_green(iid)
            return

        if self.signals[iid][cg].green == 0:
            ctrl = self.controllers[iid]
            teacher_ctrl = self._effective_controller(iid)

            if C.COLLECT_NEURAL_DATA and NEURAL_DATA_COLLECTOR is not None:
                queues = teacher_ctrl.get_queue_lengths(self.vehicles)
                avg_waits, max_waits = teacher_ctrl.get_wait_stats(self.vehicles, self.sim_time)
                downstream = teacher_ctrl.get_downstream_totals(self.vehicles)

                feats = encode_features(
                    self.cur_green[iid], queues, avg_waits, max_waits, downstream
                )
                teacher_action = teacher_ctrl.choose_next_phase(
                    self.vehicles, self.cur_green, self.sim_time
                )
                NEURAL_DATA_COLLECTOR.append(feats, teacher_action)
                chosen_phase = teacher_action
            else:
                chosen_phase = ctrl.choose_next_phase(
                    self.vehicles, self.cur_green, self.sim_time
                )

            # If the controller keeps the same phase, extend green directly.
            # Yellow is only required when switching to a different phase.
            if chosen_phase == cg:
                duration_ctrl = teacher_ctrl if (C.COLLECT_NEURAL_DATA and teacher_ctrl is not None) else ctrl
                self.signals[iid][cg].green = self._compute_green_duration(iid, duration_ctrl)
                self.signals[iid][cg].yellow = C.DEFAULT_YELLOW
                return

            self.next_green[iid] = chosen_phase
            self.cur_yellow[iid] = 1
            d_cur = C.DIRECTION_NUMS[cg]
            for lane in range(3):
                for v in self.vehicles[iid][d_cur][lane]:
                    v.stop = default_stop_pos(iid, d_cur)

    def _spawn_vehicle(self):
        iid   = RNG.randint(0, C.NO_INTERSECTIONS - 1)
        vtype = RNG.randint(0, 3)
        lane  = RNG.randint(1, 2)
        t     = RNG.randint(0, 99)
        dist  = C.DIRECTION_DIST
        if   t < dist[0]: dn = 0
        elif t < dist[1]: dn = 1
        elif t < dist[2]: dn = 2
        else:             dn = 3
        Vehicle(iid, lane,
                C.VEHICLE_TYPES[vtype],
                dn, C.DIRECTION_NUMS[dn],
                self.sim_group, self.sim_time)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def update(self, dt_s):
        if self.paused:
            return False

        self.sim_time += dt_s
        self.frame_count += 1

        self.spawn_accum_s += dt_s
        while self.spawn_accum_s >= C.SPAWN_INTERVAL:
            self.spawn_accum_s -= C.SPAWN_INTERVAL
            self._spawn_vehicle()

        for iid in range(C.NO_INTERSECTIONS):
            self.signal_accum_s[iid] += dt_s
            while self.signal_accum_s[iid] >= 1.0:
                self.signal_accum_s[iid] -= 1.0
                self._step_signals_one_second(iid)

        self.queue_accum_s += dt_s
        if self.queue_accum_s >= 0.5:
            self.queue_accum_s = 0.0
            for iid in range(C.NO_INTERSECTIONS):
                total_q = sum(
                    1
                    for d in C.DIRECTION_NUMS.values()
                    for lane in range(3)
                    for v in self.vehicles[iid][d][lane]
                    if v.crossed == 0
                )
                metrics.record_queue(iid, total_q)

        # Cull and move
        dt_scale = dt_s * C.FPS
        for v in list(self.sim_group):
            if v._offscreen:
                v.despawn()
            else:
                v.move(self.cur_green, self.cur_yellow, self.sim_time, dt_scale)
                if v._offscreen:
                    v.despawn()

        # Particles
        if C.PARTICLE_EFFECTS:
            for p in _particle_pool:
                p.update()
            _particle_pool[:] = [p for p in _particle_pool if p.alive]

        metrics.maybe_log(self.sim_time, self.vehicles)
        return bool(C.DURATION > 0 and self.sim_time >= C.DURATION)

    def build_summary(self):
        per = []
        for iid in range(C.NO_INTERSECTIONS):
            s = metrics.snapshot(iid, self.vehicles, self.sim_time)
            s["iid"] = iid
            per.append(s)
        overall = {
            "avg_wait": sum(p["avg_wait"] for p in per) / len(per),
            "avg_wait_right": sum(p["avg_wait_right"] for p in per) / len(per),
            "avg_wait_down": sum(p["avg_wait_down"] for p in per) / len(per),
            "avg_wait_left": sum(p["avg_wait_left"] for p in per) / len(per),
            "avg_wait_up": sum(p["avg_wait_up"] for p in per) / len(per),
            "p95_wait": sum(p["p95_wait"] for p in per) / len(per),
            "throughput": sum(p["throughput"] for p in per),
            "avg_queue": sum(p["avg_queue"] for p in per) / len(per),
            "fairness": sum(p["fairness"] for p in per) / len(per),
            "crossed": sum(p["crossed"] for p in per),
            "spawned": sum(p["spawned"] for p in per),
            "external_spawned": sum(p["external_spawned"] for p in per),
            "transferred_in": sum(p["transferred_in"] for p in per),
            "exited_world": sum(p["exited_world"] for p in per),
            "red_light_violations": sum(p["red_light_violations"] for p in per),
        }
        return {
            "mode": C.CONTROL_MODE,
            "seed": C.SEED,
            "spawn_rate": C.SPAWN_RATE,
            "duration_s": self.sim_time,
            "fixed_dt_s": C.FIXED_DT,
            "rl": {
                "model_path": C.RL_MODEL_PATH,
                "training": C.RL_TRAIN,
                "epsilon": (RL_POLICY.epsilon if RL_POLICY is not None else None),
                "updates": (RL_POLICY.updates if RL_POLICY is not None else 0),
            },
            "neural": {
                "model_path": C.NEURAL_MODEL_PATH,
                "collect_data": C.COLLECT_NEURAL_DATA,
                "data_path": C.NEURAL_DATA_PATH,
                "teacher_policy": C.TEACHER_POLICY,
            },
            "intersections": per,
            "overall": overall,
        }

    def _print_summary(self, summary=None):
        summary = summary or self.build_summary()
        print("\n" + "=" * 65)
        print(f"  SUMMARY  mode={C.CONTROL_MODE}  "
              f"seed={C.SEED}  spawn={C.SPAWN_RATE}")
        print("=" * 65)
        for s in summary["intersections"]:
            print(f"\n  Intersection {s['iid']}:")
            print(f"    Spawned      : {s['spawned']}")
            print(f"    External     : {s['external_spawned']}")
            print(f"    Transfer in  : {s['transferred_in']}")
            print(f"    Crossed      : {s['crossed']}")
            print(f"    Exited world : {s['exited_world']}")
            print(f"    Red violates : {s['red_light_violations']}")
            print(f"    Avg wait     : {s['avg_wait']:.2f}s")
            print(f"      by dir     : R {s['avg_wait_right']:.2f}s  "
                  f"D {s['avg_wait_down']:.2f}s  "
                  f"L {s['avg_wait_left']:.2f}s  "
                  f"U {s['avg_wait_up']:.2f}s")
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
        "rl":       (245, 90, 90),
        "neural":   (170, 110, 255),
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

        elapsed = int(self.state.sim_time)
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
        snap = metrics.snapshot(iid, self.state.vehicles, self.state.sim_time)
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
            sn  = metrics.snapshot(i, self.state.vehicles, self.state.sim_time)
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
def _finalize_run(state: SimState, artifacts: RunArtifacts):
    summary = state.build_summary()
    artifacts.write_summary(summary)
    if C.CONTROL_MODE == "rl" and RL_POLICY is not None and C.RL_TRAIN:
        RL_POLICY.save()
        print(f"[RL] saved weights to {C.RL_MODEL_PATH}")
    metrics.close()
    state._print_summary(summary)
    print(f"\n[Run artifacts] {artifacts.run_dir}")


def main():
    global metrics, RL_POLICY, NEURAL_POLICY, NEURAL_DATA_COLLECTOR
    pygame.init()
    artifacts = RunArtifacts()
    artifacts.write_config()
    metrics = MetricsEngine(artifacts)
    if C.CONTROL_MODE == "rl":
        if (not C.RL_TRAIN) and (not os.path.exists(C.RL_MODEL_PATH)):
            raise FileNotFoundError(
                f"RL model not found at {C.RL_MODEL_PATH}. "
                "Train first with --mode rl --rl-train."
            )
        RL_POLICY = TabularRLPolicy(C.RL_MODEL_PATH, training=C.RL_TRAIN)
    else:
        RL_POLICY = None

    if C.CONTROL_MODE == "neural":
        if not os.path.exists(C.NEURAL_MODEL_PATH):
            raise FileNotFoundError(
                f"Neural model not found at {C.NEURAL_MODEL_PATH}. "
                "Train first with train_neural_policy.py."
            )
        NEURAL_POLICY = load_model(C.NEURAL_MODEL_PATH)
    else:
        NEURAL_POLICY = None

    if C.COLLECT_NEURAL_DATA:
        NEURAL_DATA_COLLECTOR = NeuralDataCollector(C.NEURAL_DATA_PATH)
    else:
        NEURAL_DATA_COLLECTOR = None

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
              f"dt={C.FIXED_DT:.4f}s  out={artifacts.run_dir}")
        if C.CONTROL_MODE == "rl":
            print(f"[RL] train={C.RL_TRAIN} model={C.RL_MODEL_PATH}")
        if C.CONTROL_MODE == "neural":
            print(f"[NEURAL] model={C.NEURAL_MODEL_PATH}")
        if C.COLLECT_NEURAL_DATA:
            print(f"[NEURAL DATA] collecting to {C.NEURAL_DATA_PATH}")
            print(f"[TEACHER] policy={C.TEACHER_POLICY}")
        try:
            while True:
                done = state.update(C.FIXED_DT)
                if done:
                    print("\n[Simulation complete]")
                    break
        except KeyboardInterrupt:
            pass
        _finalize_run(state, artifacts)
        return

    renderer = Renderer(state)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _finalize_run(state, artifacts)
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
                    state._print_summary(state.build_summary())
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
        done = state.update(C.FIXED_DT)
        renderer.render(keys)
        if done:
            _finalize_run(state, artifacts)
            print("\n[Simulation complete]")
            pygame.quit()
            sys.exit(0)


if __name__ == "__main__":
    main()
