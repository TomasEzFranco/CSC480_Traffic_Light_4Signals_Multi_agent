#!/usr/bin/env python3
import argparse
import csv
import os
import statistics
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_ci95(values):
    if not values:
        return 0.0, 0.0
    m = statistics.mean(values)
    if len(values) < 2:
        return m, 0.0
    sd = statistics.stdev(values)
    ci = 1.96 * (sd / (len(values) ** 0.5))
    return m, ci


def plot_mode_bars(overall_rows, out_dir):
    metrics = ["avg_wait", "throughput", "fairness", "red_light_violations"]
    mode_order = sorted({r["mode"] for r in overall_rows})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        means, cis = [], []
        for mode in mode_order:
            vals = [float(r[metric]) for r in overall_rows if r["mode"] == mode and r[metric] != ""]
            m, ci = mean_ci95(vals)
            means.append(m)
            cis.append(ci)
        ax.bar(mode_order, means, yerr=cis, capsize=4)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "mode_comparison.png")
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_boxplots(overall_rows, out_dir):
    mode_order = sorted({r["mode"] for r in overall_rows})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric in zip(axes, ["avg_wait", "throughput"]):
        data = []
        labels = []
        for mode in mode_order:
            vals = [float(r[metric]) for r in overall_rows if r["mode"] == mode and r[metric] != ""]
            if vals:
                data.append(vals)
                labels.append(mode)
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(f"{metric} distribution")
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "seed_boxplots.png")
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] {out_path}")


def load_mode_timeseries(exp_dir, overall_rows):
    # mode -> elapsed_s -> list(value)
    queue_mode = defaultdict(lambda: defaultdict(list))

    run_to_mode = {}
    for r in overall_rows:
        run_to_mode[r["run_id"]] = r["mode"]

    for run_id, mode in run_to_mode.items():
        ts_path = os.path.join(exp_dir, run_id, "intersection_timeseries.csv")
        if not os.path.exists(ts_path):
            continue
        rows = read_csv(ts_path)

        # Aggregate each elapsed across intersections for this run
        bucket = defaultdict(list)
        for row in rows:
            elapsed = float(row["elapsed_s"])
            bucket[elapsed].append(float(row["avg_queue"]))

        for elapsed, vals in bucket.items():
            queue_mode[mode][elapsed].append(statistics.mean(vals))

    return queue_mode


def plot_timeseries_band(overall_rows, exp_dir, out_dir):
    queue_mode = load_mode_timeseries(exp_dir, overall_rows)
    if not queue_mode:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for mode in sorted(queue_mode.keys()):
        times = sorted(queue_mode[mode].keys())
        med = []
        p25 = []
        p75 = []
        for t in times:
            vals = np.array(queue_mode[mode][t], dtype=float)
            med.append(float(np.percentile(vals, 50)))
            p25.append(float(np.percentile(vals, 25)))
            p75.append(float(np.percentile(vals, 75)))

        ax.plot(times, med, label=f"{mode} median")
        ax.fill_between(times, p25, p75, alpha=0.2)

    ax.set_title("Queue Over Time (median + IQR across runs)")
    ax.set_xlabel("elapsed_s")
    ax.set_ylabel("avg_queue")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "queue_timeseries.png")
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize traffic simulation experiment results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()

    exp_dir = os.path.join(args.results_dir, args.experiment_id)
    summary_path = os.path.join(exp_dir, "summary.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary.csv: {summary_path}")

    rows = read_csv(summary_path)
    overall_rows = [r for r in rows if int(r["iid"]) == -1]

    if not overall_rows:
        raise RuntimeError("No overall rows (iid=-1) found in summary.csv")

    plot_mode_bars(overall_rows, exp_dir)
    plot_boxplots(overall_rows, exp_dir)
    plot_timeseries_band(overall_rows, exp_dir, exp_dir)


if __name__ == "__main__":
    main()
