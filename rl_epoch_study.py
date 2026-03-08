#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_str_list(raw):
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def run_cmd(cmd):
    print("[cmd]", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def run_id_for(seed, spawn):
    return f"rl_seed{seed}_spawn{spawn:.2f}".replace(".", "p")


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=0.0):
    raw = row.get(key, "")
    if raw in ("", None):
        return default
    return float(raw)


def pick_best_row(rows):
    return sorted(
        rows,
        key=lambda r: (
            as_float(r, "avg_wait"),
            as_float(r, "red_light_violations"),
            -as_float(r, "throughput"),
        ),
    )[0]


def copy_epoch_snapshots(model_template, seeds, spawns, snapshot_dir):
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for spawn in spawns:
            run_id = run_id_for(seed, spawn)
            src = Path(model_template.format(mode="rl", seed=seed, spawn=spawn, run_id=run_id))
            if not src.exists():
                raise FileNotFoundError(f"missing trained model for epoch snapshot: {src}")
            dst = snapshot_dir / f"{run_id}.json"
            shutil.copy2(src, dst)


def summarize_epoch(eval_summary_csv, epoch_idx):
    rows = read_rows(eval_summary_csv)
    rl_rows = [r for r in rows if int(float(r["iid"])) == -1 and r["mode"] == "rl"]
    if not rl_rows:
        raise RuntimeError(f"no RL overall rows in {eval_summary_csv}")

    best = pick_best_row(rl_rows)
    return {
        "epoch": epoch_idx,
        "eval_summary_csv": str(eval_summary_csv),
        "n_runs": len(rl_rows),
        "avg_wait_mean": statistics.mean(as_float(r, "avg_wait") for r in rl_rows),
        "throughput_mean": statistics.mean(as_float(r, "throughput") for r in rl_rows),
        "fairness_mean": statistics.mean(as_float(r, "fairness") for r in rl_rows),
        "red_light_violations_mean": statistics.mean(as_float(r, "red_light_violations") for r in rl_rows),
        "best_run_id": best["run_id"],
        "best_model_path": best["rl_model_path"],
        "best_avg_wait": as_float(best, "avg_wait"),
        "best_throughput": as_float(best, "throughput"),
        "best_fairness": as_float(best, "fairness"),
        "best_red_light_violations": as_float(best, "red_light_violations"),
        "rows": rl_rows,
    }


def write_epoch_csv(path, epoch_rows):
    fields = [
        "epoch", "eval_summary_csv", "n_runs",
        "avg_wait_mean", "throughput_mean", "fairness_mean", "red_light_violations_mean",
        "best_run_id", "best_model_path",
        "best_avg_wait", "best_throughput", "best_fairness", "best_red_light_violations",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in epoch_rows:
            w.writerow({k: r[k] for k in fields})


def plot_epoch_means(epoch_rows, out_path):
    epochs = [r["epoch"] for r in epoch_rows]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    metrics = [
        ("avg_wait_mean", "Avg wait (lower better)"),
        ("throughput_mean", "Throughput (higher better)"),
        ("fairness_mean", "Fairness (higher better)"),
        ("red_light_violations_mean", "Red violations (lower better)"),
    ]
    for ax, (key, title) in zip(axes.flatten(), metrics):
        vals = [float(r[key]) for r in epoch_rows]
        ax.plot(epochs, vals, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_run_lines(epoch_rows, out_path):
    series = {}
    for r in epoch_rows:
        epoch = r["epoch"]
        for row in r["rows"]:
            run_id = row["run_id"]
            series.setdefault(run_id, []).append((epoch, as_float(row, "avg_wait")))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for run_id, pts in sorted(series.items()):
        pts = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", alpha=0.9, label=run_id)
    ax.set_title("RL per-model avg_wait across epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("avg_wait (iid=-1)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] {out_path}")


def add_rl_hyperparams(cmd, args):
    if args.rl_alpha is not None:
        cmd.extend(["--rl-alpha", str(args.rl_alpha)])
    if args.rl_gamma is not None:
        cmd.extend(["--rl-gamma", str(args.rl_gamma)])
    if args.rl_epsilon is not None:
        cmd.extend(["--rl-epsilon", str(args.rl_epsilon)])
    if args.rl_epsilon_min is not None:
        cmd.extend(["--rl-epsilon-min", str(args.rl_epsilon_min)])
    if args.rl_epsilon_decay is not None:
        cmd.extend(["--rl-epsilon-decay", str(args.rl_epsilon_decay)])


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate RL across epochs, plot progress, and keep best model."
    )
    parser.add_argument("--study-id", default="rl_epoch_study_v1")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--spawn-rates", default="1.0,2.0")
    parser.add_argument("--duration-train", type=int, default=600)
    parser.add_argument("--duration-eval", type=int, default=180)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--model-template",
        default="models/rl_{seed}_{run_id}.json",
        help="Persistent model path template for training across epochs.",
    )
    parser.add_argument(
        "--best-model-output",
        default="models/rl_best_from_epoch_study.json",
        help="Best selected model copy target.",
    )
    parser.add_argument("--rl-alpha", type=float, default=None)
    parser.add_argument("--rl-gamma", type=float, default=None)
    parser.add_argument("--rl-epsilon", type=float, default=None)
    parser.add_argument("--rl-epsilon-min", type=float, default=None)
    parser.add_argument("--rl-epsilon-decay", type=float, default=None)
    parser.add_argument("--run-compare", action="store_true")
    parser.add_argument("--compare-modes", default="fixed,greedy,adaptive,random,rl")
    parser.add_argument("--compare-duration", type=int, default=180)
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    spawns = parse_float_list(args.spawn_rates)
    study_dir = Path(args.results_dir) / args.study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    archive_root = Path(args.models_dir) / "epoch_snapshots" / args.study_id
    archive_root.mkdir(parents=True, exist_ok=True)

    epoch_rows = []
    all_eval_rows = []

    for epoch in range(1, args.epochs + 1):
        ep = f"{epoch:03d}"
        train_exp = f"{args.study_id}_train_e{ep}"
        eval_exp = f"{args.study_id}_eval_e{ep}"
        snapshot_dir = archive_root / f"epoch_{ep}"
        snapshot_template = str(snapshot_dir / "{run_id}.json")

        train_cmd = [
            args.python,
            "run_experiments.py",
            "--modes", "rl",
            "--seeds", args.seeds,
            "--spawn-rates", args.spawn_rates,
            "--duration", str(args.duration_train),
            "--dt", str(args.dt),
            "--results-dir", args.results_dir,
            "--experiment-id", train_exp,
            "--rl-train",
            "--rl-model-path", args.model_template,
        ]
        if args.no_log:
            train_cmd.append("--no-log")
        add_rl_hyperparams(train_cmd, args)
        run_cmd(train_cmd)

        copy_epoch_snapshots(args.model_template, seeds, spawns, snapshot_dir)

        eval_cmd = [
            args.python,
            "run_experiments.py",
            "--modes", "rl",
            "--seeds", args.seeds,
            "--spawn-rates", args.spawn_rates,
            "--duration", str(args.duration_eval),
            "--dt", str(args.dt),
            "--results-dir", args.results_dir,
            "--experiment-id", eval_exp,
            "--rl-model-path", snapshot_template,
        ]
        if args.no_log:
            eval_cmd.append("--no-log")
        add_rl_hyperparams(eval_cmd, args)
        run_cmd(eval_cmd)

        eval_summary = Path(args.results_dir) / eval_exp / "summary.csv"
        summary = summarize_epoch(eval_summary, epoch)
        epoch_rows.append(summary)
        for row in summary["rows"]:
            all_eval_rows.append({
                "epoch": epoch,
                **row,
            })
        print(
            "[epoch]",
            epoch,
            f"mean_avg_wait={summary['avg_wait_mean']:.4f}",
            f"mean_throughput={summary['throughput_mean']:.4f}",
            f"mean_red={summary['red_light_violations_mean']:.4f}",
            f"best_run={summary['best_run_id']}",
        )

    epoch_csv = study_dir / "epoch_summary.csv"
    write_epoch_csv(epoch_csv, epoch_rows)
    print(f"[out] {epoch_csv}")

    all_rows_path = study_dir / "all_eval_rows.csv"
    if all_eval_rows:
        fields = ["epoch"] + list(all_eval_rows[0].keys())[1:]
        with open(all_rows_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_eval_rows)
        print(f"[out] {all_rows_path}")

    plot_epoch_means(epoch_rows, study_dir / "rl_epoch_metrics.png")
    plot_run_lines(epoch_rows, study_dir / "rl_epoch_run_comparison.png")

    best_row = pick_best_row(all_eval_rows)
    best_src = Path(best_row["rl_model_path"])
    if not best_src.exists():
        raise FileNotFoundError(f"best model file missing: {best_src}")
    best_dst = Path(args.best_model_output)
    best_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_src, best_dst)

    best_meta = {
        "study_id": args.study_id,
        "selected_epoch": int(best_row["epoch"]),
        "selected_run_id": best_row["run_id"],
        "selected_model_path": str(best_src),
        "selected_metrics": {
            "avg_wait": as_float(best_row, "avg_wait"),
            "throughput": as_float(best_row, "throughput"),
            "fairness": as_float(best_row, "fairness"),
            "red_light_violations": as_float(best_row, "red_light_violations"),
        },
        "best_model_output": str(best_dst),
        "selection_rule": [
            "lowest avg_wait",
            "then lowest red_light_violations",
            "then highest throughput",
        ],
    }
    best_meta_path = study_dir / "best_rl_selection.json"
    with open(best_meta_path, "w", encoding="utf-8") as f:
        json.dump(best_meta, f, indent=2)
    print(f"[out] {best_meta_path}")
    print(f"[best] copied model -> {best_dst}")

    if args.run_compare:
        compare_exp = f"{args.study_id}_compare_best"
        compare_cmd = [
            args.python,
            "run_experiments.py",
            "--modes", args.compare_modes,
            "--seeds", args.seeds,
            "--spawn-rates", args.spawn_rates,
            "--duration", str(args.compare_duration),
            "--dt", str(args.dt),
            "--results-dir", args.results_dir,
            "--experiment-id", compare_exp,
            "--rl-model-path", str(best_dst),
        ]
        if args.no_log:
            compare_cmd.append("--no-log")
        run_cmd(compare_cmd)
        run_cmd([
            args.python,
            "visualize_results.py",
            "--results-dir", args.results_dir,
            "--experiment-id", compare_exp,
        ])


if __name__ == "__main__":
    main()
