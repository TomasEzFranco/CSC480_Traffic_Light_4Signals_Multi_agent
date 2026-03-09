import argparse
import os
import subprocess
import sys


def parse_list(raw):
    return [x.strip() for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Collect supervised neural training data from a configurable teacher policy")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", default="21,22,23,24,25,26,27,28,29,30")
    parser.add_argument("--spawn-rates", default="1.0")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--data-path", default="data/neural/hybrid_train.csv")
    parser.add_argument("--teacher-policy", default="hybrid", choices=["greedy", "hybrid"])
    args = parser.parse_args()

    seeds = [int(x) for x in parse_list(args.seeds)]
    spawns = [float(x) for x in parse_list(args.spawn_rates)]

    if os.path.exists(args.data_path):
        os.remove(args.data_path)

    for seed in seeds:
        for spawn in spawns:
            cmd = [
                args.python,
                "simulation.py",
                "--headless",
                "--mode", "greedy",
                "--seed", str(seed),
                "--spawn-rate", str(spawn),
                "--duration", str(args.duration),
                "--dt", str(args.dt),
                "--collect-neural-data",
                "--neural-data-path", args.data_path,
                "--teacher-policy", args.teacher_policy,
                "--no-log",
            ]
            print("RUN:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
