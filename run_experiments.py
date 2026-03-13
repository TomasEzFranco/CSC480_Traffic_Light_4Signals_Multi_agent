#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_str_list(raw):
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def run_one(py_cmd, mode, seed, spawn_rate, duration, dt, results_dir, exp_id,
            no_log, rl_model_path, rl_train, rl_alpha, rl_gamma,
            rl_epsilon, rl_epsilon_min, rl_epsilon_decay,
            rl_starvation_t, rl_w_queue_delta, rl_w_wait_delta, rl_w_maxwait_delta,
            rl_w_throughput, rl_w_cur_queue, rl_w_cur_wait_mass, rl_w_cur_maxwait,
            rl_w_imbalance, rl_w_starved, rl_switch_penalty,
            fixed_green_seconds, shared_green_seconds,
            neural_model_path):
    run_id = f"{mode}_seed{seed}_spawn{spawn_rate:.2f}".replace(".", "p")
    resolved_rl_model_path = rl_model_path.format(
        mode=mode,
        seed=seed,
        spawn=spawn_rate,
        run_id=run_id,
    )
    cmd = [
        py_cmd,
        "simulation.py",
        "--headless",
        "--mode", mode,
        "--seed", str(seed),
        "--spawn-rate", str(spawn_rate),
        "--duration", str(duration),
        "--dt", str(dt),
        "--fixed-green-seconds", str(fixed_green_seconds),
        "--shared-green-seconds", str(shared_green_seconds),
        "--results-dir", results_dir,
        "--experiment-id", exp_id,
        "--run-id", run_id,
    ]
    if mode == "rl":
        cmd.extend(["--rl-model-path", resolved_rl_model_path])
        if rl_train:
            cmd.append("--rl-train")
        if rl_alpha is not None:
            cmd.extend(["--rl-alpha", str(rl_alpha)])
        if rl_gamma is not None:
            cmd.extend(["--rl-gamma", str(rl_gamma)])
        if rl_epsilon is not None:
            cmd.extend(["--rl-epsilon", str(rl_epsilon)])
        if rl_epsilon_min is not None:
            cmd.extend(["--rl-epsilon-min", str(rl_epsilon_min)])
        if rl_epsilon_decay is not None:
            cmd.extend(["--rl-epsilon-decay", str(rl_epsilon_decay)])
        if rl_starvation_t is not None:
            cmd.extend(["--rl-starvation-t", str(rl_starvation_t)])
        if rl_w_queue_delta is not None:
            cmd.extend(["--rl-w-queue-delta", str(rl_w_queue_delta)])
        if rl_w_wait_delta is not None:
            cmd.extend(["--rl-w-wait-delta", str(rl_w_wait_delta)])
        if rl_w_maxwait_delta is not None:
            cmd.extend(["--rl-w-maxwait-delta", str(rl_w_maxwait_delta)])
        if rl_w_throughput is not None:
            cmd.extend(["--rl-w-throughput", str(rl_w_throughput)])
        if rl_w_cur_queue is not None:
            cmd.extend(["--rl-w-cur-queue", str(rl_w_cur_queue)])
        if rl_w_cur_wait_mass is not None:
            cmd.extend(["--rl-w-cur-wait-mass", str(rl_w_cur_wait_mass)])
        if rl_w_cur_maxwait is not None:
            cmd.extend(["--rl-w-cur-maxwait", str(rl_w_cur_maxwait)])
        if rl_w_imbalance is not None:
            cmd.extend(["--rl-w-imbalance", str(rl_w_imbalance)])
        if rl_w_starved is not None:
            cmd.extend(["--rl-w-starved", str(rl_w_starved)])
        if rl_switch_penalty is not None:
            cmd.extend(["--rl-switch-penalty", str(rl_switch_penalty)])
    if mode == "neural":
        cmd.extend(["--neural-model-path", neural_model_path])
    if no_log:
        cmd.append("--no-log")

    print(f"[RUN] mode={mode:8s} seed={seed:<4d} spawn={spawn_rate:<4.2f} run_id={run_id}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        print("[ERROR] simulation failed")
        if proc.stdout:
            print(proc.stdout[-2000:])
        if proc.stderr:
            print(proc.stderr[-2000:])
        raise RuntimeError(f"run failed: {run_id}")

    summary_path = os.path.join(results_dir, exp_id, run_id, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"missing summary.json: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return run_id, summary


def flatten_summary(exp_id, run_id, summary):
    rows = []
    base = {
        "experiment_id": exp_id,
        "run_id": run_id,
        "mode": summary["mode"],
        "seed": summary["seed"],
        "spawn_rate": summary["spawn_rate"],
        "duration_s": summary["duration_s"],
        "fixed_dt_s": summary["fixed_dt_s"],
        "rl_model_path": summary.get("rl", {}).get("model_path", ""),
        "rl_training": summary.get("rl", {}).get("training", False),
    }

    for s in summary["intersections"]:
        rows.append({
            **base,
            "iid": s["iid"],
            "avg_wait": s["avg_wait"],
            "avg_wait_right": s["avg_wait_right"],
            "avg_wait_down": s["avg_wait_down"],
            "avg_wait_left": s["avg_wait_left"],
            "avg_wait_up": s["avg_wait_up"],
            "p95_wait": s["p95_wait"],
            "max_wait": s["max_wait"],
            "throughput": s["throughput"],
            "avg_queue": s["avg_queue"],
            "max_queue": s["max_queue"],
            "fairness": s["fairness"],
            "crossed": s["crossed"],
            "spawned": s["spawned"],
            "external_spawned": s["external_spawned"],
            "transferred_in": s["transferred_in"],
            "exited_world": s["exited_world"],
            "red_light_violations": s["red_light_violations"],
        })

    o = summary["overall"]
    rows.append({
        **base,
        "iid": -1,
        "avg_wait": o["avg_wait"],
        "avg_wait_right": o["avg_wait_right"],
        "avg_wait_down": o["avg_wait_down"],
        "avg_wait_left": o["avg_wait_left"],
        "avg_wait_up": o["avg_wait_up"],
        "p95_wait": o["p95_wait"],
        "max_wait": "",
        "throughput": o["throughput"],
        "avg_queue": o["avg_queue"],
        "max_queue": "",
        "fairness": o["fairness"],
        "crossed": o["crossed"],
        "spawned": o["spawned"],
        "external_spawned": o["external_spawned"],
        "transferred_in": o["transferred_in"],
        "exited_world": o["exited_world"],
        "red_light_violations": o["red_light_violations"],
    })
    return rows


def write_summary_csv(path, rows):
    if not rows:
        return
    fields = [
        "experiment_id", "run_id", "mode", "seed", "spawn_rate",
        "duration_s", "fixed_dt_s", "rl_model_path", "rl_training",
        "iid", "avg_wait",
        "avg_wait_right", "avg_wait_down", "avg_wait_left", "avg_wait_up",
        "p95_wait",
        "max_wait", "throughput", "avg_queue", "max_queue", "fairness",
        "crossed", "spawned", "external_spawned", "transferred_in",
        "exited_world", "red_light_violations",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _as_float(row, key, default=0.0):
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    return float(raw)


def pick_best_rl_row(summary_csv):
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"missing eval summary: {summary_csv}")

    candidates = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                iid = int(float(r.get("iid", "")))
            except ValueError:
                continue
            if iid != -1:
                continue
            if r.get("mode") != "rl":
                continue
            if not r.get("rl_model_path"):
                continue
            candidates.append(r)

    if not candidates:
        raise RuntimeError(f"no RL overall rows found in {summary_csv}")

    return sorted(
        candidates,
        key=lambda r: (
            _as_float(r, "avg_wait"),
            _as_float(r, "red_light_violations"),
            -_as_float(r, "throughput"),
        ),
    )[0]


def select_best_rl_model(results_dir, eval_experiment_id, output_model_path):
    summary_csv = os.path.join(results_dir, eval_experiment_id, "summary.csv")
    best = pick_best_rl_row(summary_csv)

    model_src = Path(best["rl_model_path"])
    if not model_src.exists():
        raise FileNotFoundError(f"best model path does not exist: {model_src}")

    model_dst = Path(output_model_path)
    model_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_src, model_dst)

    meta = {
        "source_eval_experiment": eval_experiment_id,
        "selected_run_id": best["run_id"],
        "selected_model_path": str(model_src),
        "selected_metrics": {
            "avg_wait": _as_float(best, "avg_wait"),
            "throughput": _as_float(best, "throughput"),
            "red_light_violations": _as_float(best, "red_light_violations"),
        },
        "best_model_path": str(model_dst),
    }
    return meta


def main():
    parser = argparse.ArgumentParser(description="Run experiment matrix for traffic simulation")
    parser.add_argument("--modes", default="fixed,greedy,adaptive,random")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--spawn-rates", default="1.0,2.0")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--fixed-green-seconds", type=int, default=3)
    parser.add_argument("--shared-green-seconds", type=int, default=3)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--rl-model-path",
        default="models/rl_qtable.json",
        help="Path for RL weights. Supports {mode},{seed},{spawn},{run_id} templates.",
    )
    parser.add_argument("--rl-train", action="store_true")
    parser.add_argument("--rl-alpha", type=float, default=None)
    parser.add_argument("--rl-gamma", type=float, default=None)
    parser.add_argument("--rl-epsilon", type=float, default=None)
    parser.add_argument("--rl-epsilon-min", type=float, default=None)
    parser.add_argument("--rl-epsilon-decay", type=float, default=None)
    parser.add_argument("--rl-starvation-t", type=float, default=None)
    parser.add_argument("--rl-w-throughput", type=float, default=None)
    parser.add_argument("--rl-w-wait-delta", type=float, default=None)
    parser.add_argument("--rl-switch-penalty", type=float, default=None)
    # Legacy reward knobs retained only for backward CLI compatibility.
    parser.add_argument("--rl-w-maxwait-delta", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rl-w-queue-delta", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rl-w-cur-queue", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rl-w-cur-wait-mass", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rl-w-cur-maxwait", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rl-w-imbalance", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rl-w-starved", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--neural-model-path", default="models/neural_hybrid.pt")
    parser.add_argument(
        "--select-best-rl-from",
        "--auto-best-rl-from-experiment",
        dest="select_best_rl_from",
        default="",
        help=(
            "Pick best RL model from results/<experiment-id>/summary.csv "
            "(lowest avg_wait, then fewer red_light_violations, then higher throughput)."
        ),
    )
    parser.add_argument(
        "--select-best-rl-output",
        "--auto-best-rl-output",
        dest="select_best_rl_output",
        default="models/rl_best.json",
        help="Where to copy selected best RL model.",
    )
    args = parser.parse_args()

    modes = parse_str_list(args.modes)
    seeds = parse_int_list(args.seeds)
    spawn_rates = parse_float_list(args.spawn_rates)

    if "rl" in modes and args.rl_train and args.select_best_rl_from.strip():
        raise ValueError("--select-best-rl-from cannot be used with --rl-train for RL runs")

    exp_id = args.experiment_id.strip() or datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.results_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    selected_meta = None
    if args.select_best_rl_from.strip():
        selected_meta = select_best_rl_model(
            args.results_dir,
            args.select_best_rl_from.strip(),
            args.select_best_rl_output,
        )
        selection_path = os.path.join(exp_dir, "best_rl_selection.json")
        with open(selection_path, "w", encoding="utf-8") as f:
            json.dump(selected_meta, f, indent=2)

        print(
            "[BEST RL]",
            f"run_id={selected_meta['selected_run_id']}",
            f"model={selected_meta['selected_model_path']}",
            f"avg_wait={selected_meta['selected_metrics']['avg_wait']:.4f}",
            f"throughput={selected_meta['selected_metrics']['throughput']:.4f}",
            f"viol={selected_meta['selected_metrics']['red_light_violations']:.0f}",
        )
        print(f"[BEST RL] copied to {selected_meta['best_model_path']}")
        print(f"[BEST RL] metadata: {selection_path}")

        if "rl" in modes:
            args.rl_model_path = str(Path(args.select_best_rl_output))
            print(f"[BEST RL] RL runs will use: {args.rl_model_path}")

    all_rows = []
    total = len(modes) * len(seeds) * len(spawn_rates)
    idx = 0

    for mode, seed, spawn in itertools.product(modes, seeds, spawn_rates):
        idx += 1
        print(f"\n[{idx}/{total}] starting")
        run_id, summary = run_one(
            args.python,
            mode,
            seed,
            spawn,
            args.duration,
            args.dt,
            args.results_dir,
            exp_id,
            args.no_log,
            args.rl_model_path,
            args.rl_train,
            args.rl_alpha,
            args.rl_gamma,
            args.rl_epsilon,
            args.rl_epsilon_min,
            args.rl_epsilon_decay,
            args.rl_starvation_t,
            args.rl_w_queue_delta,
            args.rl_w_wait_delta,
            args.rl_w_maxwait_delta,
            args.rl_w_throughput,
            args.rl_w_cur_queue,
            args.rl_w_cur_wait_mass,
            args.rl_w_cur_maxwait,
            args.rl_w_imbalance,
            args.rl_w_starved,
            args.rl_switch_penalty,
            args.fixed_green_seconds,
            args.shared_green_seconds,
            args.neural_model_path,
        )
        all_rows.extend(flatten_summary(exp_id, run_id, summary))

    summary_csv = os.path.join(exp_dir, "summary.csv")
    write_summary_csv(summary_csv, all_rows)
    print(f"\n[Done] wrote {summary_csv}")


if __name__ == "__main__":
    main()
