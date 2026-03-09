
import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def ensure_columns(df):
    # Backward compatible renames if needed
    rename_map = {}
    for old, new in [
        ("avg_wait_s", "avg_wait"),
        ("p95_wait_s", "p95_wait"),
        ("throughput_per_min", "throughput"),
    ]:
        if old in df.columns and new not in df.columns:
            rename_map[old] = new
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def main():
    parser = argparse.ArgumentParser(description="Summarize and plot traffic experiment results.")
    parser.add_argument("--summary-csv", required=True, help="Path to summary.csv produced by run_experiments.py")
    parser.add_argument("--out-dir", default="analysis_plots", help="Where to save tables and plots")
    parser.add_argument("--title-prefix", default="", help="Optional title prefix for figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    df = ensure_columns(df)

    # Keep overall rows only
    if "iid" in df.columns:
        overall = df[df["iid"].astype(str) == "-1"].copy()
        if overall.empty:
            overall = df.copy()
    else:
        overall = df.copy()

    numeric_cols = ["avg_wait", "p95_wait", "throughput", "avg_queue", "fairness"]
    for c in numeric_cols:
        overall[c] = pd.to_numeric(overall[c], errors="coerce")

    # Save raw overall rows
    overall.to_csv(out_dir / "overall_rows.csv", index=False)

    mean_df = overall.groupby("mode", as_index=False)[numeric_cols].mean()
    std_df = overall.groupby("mode", as_index=False)[numeric_cols].std().fillna(0.0)

    mean_df.to_csv(out_dir / "mode_means.csv", index=False)
    std_df.to_csv(out_dir / "mode_stds.csv", index=False)

    # Nice ranking tables
    for metric, ascending in [
        ("avg_wait", True),
        ("p95_wait", True),
        ("throughput", False),
        ("avg_queue", True),
        ("fairness", False),
    ]:
        ranked = mean_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
        ranked.to_csv(out_dir / f"ranking_{metric}.csv", index=False)

    title_prefix = (args.title_prefix + " ").strip()

    def plot_metric(metric, ylabel, ascending=True):
        ranked = mean_df.sort_values(metric, ascending=ascending)
        plt.figure(figsize=(8, 5))
        plt.bar(ranked["mode"], ranked[metric])
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix}{metric} by mode")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_bar.png", dpi=180)
        plt.close()

    plot_metric("avg_wait", "Average wait (s)", ascending=True)
    plot_metric("p95_wait", "P95 wait (s)", ascending=True)
    plot_metric("throughput", "Throughput (veh/min)", ascending=False)
    plot_metric("avg_queue", "Average queue", ascending=True)
    plot_metric("fairness", "Fairness", ascending=False)

    # Per-seed line charts if seed exists
    if "seed" in overall.columns:
        overall["seed"] = pd.to_numeric(overall["seed"], errors="coerce")
        for metric, ylabel in [
            ("avg_wait", "Average wait (s)"),
            ("p95_wait", "P95 wait (s)"),
            ("throughput", "Throughput (veh/min)"),
            ("avg_queue", "Average queue"),
        ]:
            plt.figure(figsize=(8, 5))
            for mode, g in overall.groupby("mode"):
                g = g.sort_values("seed")
                plt.plot(g["seed"], g[metric], marker="o", label=mode)
            plt.xlabel("Seed")
            plt.ylabel(ylabel)
            plt.title(f"{title_prefix}{metric} across seeds")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{metric}_by_seed.png", dpi=180)
            plt.close()

    print(f"Saved analysis to: {out_dir}")

if __name__ == "__main__":
    main()
