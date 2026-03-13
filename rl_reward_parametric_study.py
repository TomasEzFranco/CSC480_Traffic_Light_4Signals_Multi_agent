#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row, key, default):
    val = row.get(key, "")
    if val in ("", None):
        return float(default)
    return float(val)


def run_cmd(cmd):
    print("[cmd]", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def run_config(args, config):
    cfg_id = f"{args.study_id}_{config['label']}"
    model_path = f"models/studies/{args.study_id}/{config['label']}_shared.json"
    best_out = f"models/studies/{args.study_id}/{config['label']}_best.json"

    cmd = [
        args.python,
        "rl_epoch_study.py",
        "--study-id", cfg_id,
        "--results-dir", args.results_dir,
        "--epochs", str(args.epochs),
        "--train-seeds", args.train_seeds,
        "--val-seeds", args.val_seeds,
        "--test-seeds", args.test_seeds,
        "--spawn-rates", args.spawn_rates,
        "--duration-train", str(args.duration_train),
        "--duration-eval", str(args.duration_eval),
        "--dt", str(args.dt),
        "--model-path", model_path,
        "--best-model-output", best_out,
        "--rl-alpha", str(args.rl_alpha),
        "--rl-gamma", str(args.rl_gamma),
        "--rl-epsilon", str(args.rl_epsilon),
        "--rl-epsilon-min", str(args.rl_epsilon_min),
        "--rl-epsilon-decay", str(args.rl_epsilon_decay),
        "--rl-state-profile", args.rl_state_profile,
        "--rl-w-throughput", str(config["rl_w_throughput"]),
        "--rl-w-wait-delta", str(config["rl_w_wait_delta"]),
        "--rl-switch-penalty", str(config["rl_switch_penalty"]),
    ]
    if args.no_log:
        cmd.append("--no-log")

    run_cmd(cmd)
    summary_csv = Path(args.results_dir) / cfg_id / "epoch_summary.csv"
    rows = read_csv_rows(summary_csv)
    parsed = []
    for row in rows:
        parsed.append({
            "study_id": cfg_id,
            "label": config["label"],
            "legend": config["legend"],
            "rl_w_throughput": config["rl_w_throughput"],
            "rl_w_wait_delta": config["rl_w_wait_delta"],
            "rl_switch_penalty": config["rl_switch_penalty"],
            "epoch": int(float(row["epoch"])),
            "avg_wait_mean": float(row["avg_wait_mean"]),
            "throughput_mean": float(row["throughput_mean"]),
            "fairness_mean": float(row["fairness_mean"]),
            "red_light_violations_mean": float(row["red_light_violations_mean"]),
            "best_model_path": row["best_model_path"],
        })
    return parsed


def write_combined_csv(path, rows):
    fields = [
        "study_id",
        "label",
        "legend",
        "rl_w_throughput",
        "rl_w_wait_delta",
        "rl_switch_penalty",
        "epoch",
        "avg_wait_mean",
        "throughput_mean",
        "fairness_mean",
        "red_light_violations_mean",
        "best_model_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def select_best_epoch_rows(rows):
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    best_rows = []
    for label, vals in by_label.items():
        vals_sorted = sorted(
            vals,
            key=lambda r: (
                r["avg_wait_mean"],
                r["red_light_violations_mean"],
                -r["throughput_mean"],
            ),
        )
        best_rows.append(vals_sorted[0])
    return sorted(best_rows, key=lambda r: r["avg_wait_mean"])


def write_best_csv(path, rows):
    fields = [
        "study_id",
        "label",
        "legend",
        "rl_w_throughput",
        "rl_w_wait_delta",
        "rl_switch_penalty",
        "epoch",
        "avg_wait_mean",
        "throughput_mean",
        "fairness_mean",
        "red_light_violations_mean",
        "best_model_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _group_by_label(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    for vals in grouped.values():
        vals.sort(key=lambda r: r["epoch"])
    return grouped


def _style_cycle():
    return [
        ("#1f77b4", "o"),
        ("#d62728", "s"),
        ("#2ca02c", "^"),
        ("#ff7f0e", "D"),
        ("#9467bd", "P"),
        ("#17becf", "X"),
    ]


def plot_epoch_lines(rows, out_path):
    grouped = _group_by_label(rows)
    styles = _style_cycle()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    panels = [
        ("avg_wait_mean", "Validation Avg Wait by Epoch", "Average wait (s)", "lower is better"),
        ("throughput_mean", "Validation Throughput by Epoch", "Throughput (veh/min)", "higher is better"),
        ("fairness_mean", "Validation Fairness by Epoch", "Fairness", "higher is better"),
        ("red_light_violations_mean", "Validation Red-Light Violations by Epoch", "Violations", "lower is better"),
    ]
    for ax, (metric, title, ylab, subtitle) in zip(axes.flatten(), panels):
        for idx, label in enumerate(sorted(grouped.keys())):
            color, marker = styles[idx % len(styles)]
            vals = grouped[label]
            xs = [r["epoch"] for r in vals]
            ys = [r[metric] for r in vals]
            legend = vals[0]["legend"]
            ax.plot(
                xs,
                ys,
                label=legend,
                color=color,
                marker=marker,
                linewidth=2.4,
                markersize=6,
                alpha=0.95,
            )
        ax.set_title(f"{title}\n({subtitle})", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.28)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, fontsize=9)
    fig.suptitle("RL Reward Parametric Study (Throughput/Wait Delta vs Switch Penalty)", fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0.0, 1, 0.92])
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_best_tradeoff(best_rows, out_path):
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    styles = _style_cycle()
    for idx, row in enumerate(best_rows):
        color, marker = styles[idx % len(styles)]
        x = row["avg_wait_mean"]
        y = row["throughput_mean"]
        ax.scatter(x, y, s=140, marker=marker, color=color, edgecolor="black", linewidth=0.8, alpha=0.9)
        ax.text(x + 0.06, y + 0.12, row["legend"], fontsize=9)
    ax.set_title("Best-Epoch Tradeoff per Reward Configuration")
    ax.set_xlabel("Avg wait (s)  [lower is better]")
    ax.set_ylabel("Throughput (veh/min)  [higher is better]")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[plot] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RL reward-parameter sweep and produce comparison plots."
    )
    parser.add_argument("--study-id", default="rl_reward_sweep_v1")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-seeds", default="42,43,44,45")
    parser.add_argument("--val-seeds", default="48,49")
    parser.add_argument("--test-seeds", default="50")
    parser.add_argument("--spawn-rates", default="1.0")
    parser.add_argument("--duration-train", type=int, default=90)
    parser.add_argument("--duration-eval", type=int, default=60)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--rl-alpha", type=float, default=0.2)
    parser.add_argument("--rl-gamma", type=float, default=0.95)
    parser.add_argument("--rl-epsilon", type=float, default=0.20)
    parser.add_argument("--rl-epsilon-min", type=float, default=0.02)
    parser.add_argument("--rl-epsilon-decay", type=float, default=0.9995)
    parser.add_argument(
        "--rl-state-profile",
        choices=["coarse", "default", "fine"],
        default="fine",
        help="RL state bucket profile to use during each reward-config run.",
    )
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip rerunning experiments and build final ranking/plots from combined_epoch_metrics.csv.",
    )
    args = parser.parse_args()

    configs = [
        {
            "label": "baseline",
            "legend": "Baseline  TP=1.00  Wd=0.050  SW=0.20",
            "rl_w_throughput": 1.00,
            "rl_w_wait_delta": 0.050,
            "rl_switch_penalty": 0.20,
        },
        {
            "label": "queue_focus",
            "legend": "Throughput focus  TP=1.25  Wd=0.040  SW=0.20",
            "rl_w_throughput": 1.25,
            "rl_w_wait_delta": 0.040,
            "rl_switch_penalty": 0.20,
        },
        {
            "label": "wait_focus",
            "legend": "Wait focus  TP=0.75  Wd=0.070  SW=0.20",
            "rl_w_throughput": 0.75,
            "rl_w_wait_delta": 0.070,
            "rl_switch_penalty": 0.20,
        },
        {
            "label": "low_switch_penalty",
            "legend": "Low switch cost  TP=1.00  Wd=0.050  SW=0.10",
            "rl_w_throughput": 1.00,
            "rl_w_wait_delta": 0.050,
            "rl_switch_penalty": 0.10,
        },
    ]

    out_dir = Path(args.results_dir) / args.study_id
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = out_dir / "combined_epoch_metrics.csv"

    if args.reuse_existing:
        all_rows = []
        for row in read_csv_rows(combined_csv):
            all_rows.append({
                "study_id": row["study_id"],
                "label": row["label"],
                "legend": row["legend"],
                "rl_w_throughput": _f(row, "rl_w_throughput", _f(row, "rl_w_queue_delta", 1.00)),
                "rl_w_wait_delta": _f(row, "rl_w_wait_delta", 0.050),
                "rl_switch_penalty": _f(row, "rl_switch_penalty", 0.20),
                "epoch": int(float(row["epoch"])),
                "avg_wait_mean": float(row["avg_wait_mean"]),
                "throughput_mean": float(row["throughput_mean"]),
                "fairness_mean": float(row["fairness_mean"]),
                "red_light_violations_mean": float(row["red_light_violations_mean"]),
                "best_model_path": row["best_model_path"],
            })
        print(f"[reuse] loaded {len(all_rows)} rows from {combined_csv}")
    else:
        all_rows = []
        for cfg in configs:
            print(
                f"[config] {cfg['legend']} "
                f"(w_throughput={cfg['rl_w_throughput']}, w_wait_delta={cfg['rl_w_wait_delta']}, switch_penalty={cfg['rl_switch_penalty']})"
            )
            all_rows.extend(run_config(args, cfg))
        write_combined_csv(combined_csv, all_rows)
        print(f"[out] {combined_csv}")

    best_rows = select_best_epoch_rows(all_rows)
    best_csv = out_dir / "best_config_by_validation.csv"
    write_best_csv(best_csv, best_rows)
    print(f"[out] {best_csv}")

    plot_epoch_lines(all_rows, out_dir / "reward_sweep_epoch_dashboard.png")
    plot_best_tradeoff(best_rows, out_dir / "reward_sweep_best_tradeoff.png")


if __name__ == "__main__":
    main()
