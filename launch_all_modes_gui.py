import os
import sys
import time
import subprocess
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
SIM_PATH = REPO_DIR / "simulation.py"

PYTHON_EXE = sys.executable


SEED = "42"
SPAWN_RATE = "1.5"
DT = "0.0166667"


DURATION = "0"


WINDOW_POSITIONS = {
    "fixed":    (0, 0),
    "adaptive": (1450, 0),
    "greedy":   (0, 850),
    "random":   (1450, 850),
    "rl":       (0, 1700),
    "neural":   (1450, 1700),
}


MODES = ["fixed", "adaptive", "greedy", "random", "rl", "neural"]


RL_MODEL = REPO_DIR / "models" / "rl_boost_best.json"
NEURAL_MODEL = REPO_DIR / "models" / "neural_hybrid.pt"


def build_command(mode: str):
    cmd = [
        str(PYTHON_EXE),
        str(SIM_PATH),
        "--mode", mode,
        "--seed", SEED,
        "--spawn-rate", SPAWN_RATE,
        "--duration", DURATION,
        "--dt", DT,
    ]

    if mode == "rl":
        cmd += ["--rl-model-path", str(RL_MODEL)]

    if mode == "neural":
        cmd += ["--neural-model-path", str(NEURAL_MODEL)]

    return cmd


def launch_mode(mode: str):
    env = os.environ.copy()

    x, y = WINDOW_POSITIONS.get(mode, (100, 100))
    env["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"

    cmd = build_command(mode)

    print(f"Launching {mode}: {' '.join(cmd)}")

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE

    return subprocess.Popen(
        cmd,
        cwd=str(REPO_DIR),
        env=env,
        creationflags=creationflags,
    )


def main():
    if not SIM_PATH.exists():
        raise FileNotFoundError(f"simulation.py not found at: {SIM_PATH}")

    if "rl" in MODES and not RL_MODEL.exists():
        raise FileNotFoundError(f"RL model not found: {RL_MODEL}")

    if "neural" in MODES and not NEURAL_MODEL.exists():
        raise FileNotFoundError(f"Neural model not found: {NEURAL_MODEL}")

    procs = []

    for mode in MODES:
        p = launch_mode(mode)
        procs.append((mode, p))
        time.sleep(1.0)

    print("\nAll simulator windows launched.")
    print("Close each simulator window manually when you are done recording.\n")

    try:
        while True:
            alive = [p.poll() is None for _, p in procs]
            if not any(alive):
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nLauncher interrupted. Closing child processes...")
        for mode, p in procs:
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()