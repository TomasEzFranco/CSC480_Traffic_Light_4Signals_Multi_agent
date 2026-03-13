#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def add_optional(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal wrapper around rl_epoch_study.py with stable defaults "
            "for train/val/test splits and 3-factor RL reward."
        )
    )
    parser.add_argument("--study-id", default="rl_simple_40_53_v1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-seeds", default="40,41,42,43,44,45,46,47")
    parser.add_argument("--val-seeds", default="48,49,50")
    parser.add_argument("--test-seeds", default="51,52,53")
    parser.add_argument("--spawn-rates", default="2.0")
    parser.add_argument("--duration-train", type=int, default=300)
    parser.add_argument("--duration-eval", type=int, default=180)
    parser.add_argument("--compare-duration", type=int, default=180)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--compare-modes", default="fixed,greedy,adaptive,random,rl")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--best-model-output", default="")
    parser.add_argument("--rl-alpha", type=float, default=None)
    parser.add_argument("--rl-gamma", type=float, default=None)
    parser.add_argument("--rl-epsilon", type=float, default=None)
    parser.add_argument("--rl-epsilon-min", type=float, default=None)
    parser.add_argument("--rl-epsilon-decay", type=float, default=None)
    parser.add_argument("--rl-w-throughput", type=float, default=1.00)
    parser.add_argument("--rl-w-wait-delta", type=float, default=0.050)
    parser.add_argument("--rl-switch-penalty", type=float, default=0.20)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    study_id = args.study_id.strip()
    if not study_id:
        raise ValueError("--study-id cannot be empty")

    model_path = args.model_path.strip() or f"{args.models_dir}/studies/{study_id}/shared_qtable.json"
    best_model_output = args.best_model_output.strip() or f"{args.models_dir}/rl_best_{study_id}.json"

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(best_model_output).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        "rl_epoch_study.py",
        "--study-id",
        study_id,
        "--results-dir",
        args.results_dir,
        "--models-dir",
        args.models_dir,
        "--epochs",
        str(args.epochs),
        "--train-seeds",
        args.train_seeds,
        "--val-seeds",
        args.val_seeds,
        "--test-seeds",
        args.test_seeds,
        "--spawn-rates",
        args.spawn_rates,
        "--duration-train",
        str(args.duration_train),
        "--duration-eval",
        str(args.duration_eval),
        "--compare-duration",
        str(args.compare_duration),
        "--dt",
        str(args.dt),
        "--compare-modes",
        args.compare_modes,
        "--model-path",
        model_path,
        "--best-model-output",
        best_model_output,
        "--rl-w-throughput",
        str(args.rl_w_throughput),
        "--rl-w-wait-delta",
        str(args.rl_w_wait_delta),
        "--rl-switch-penalty",
        str(args.rl_switch_penalty),
    ]
    add_optional(cmd, "--rl-alpha", args.rl_alpha)
    add_optional(cmd, "--rl-gamma", args.rl_gamma)
    add_optional(cmd, "--rl-epsilon", args.rl_epsilon)
    add_optional(cmd, "--rl-epsilon-min", args.rl_epsilon_min)
    add_optional(cmd, "--rl-epsilon-decay", args.rl_epsilon_decay)
    if args.no_log:
        cmd.append("--no-log")
    if not args.skip_compare:
        cmd.append("--run-compare")

    print("[simple-rl] launching:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
