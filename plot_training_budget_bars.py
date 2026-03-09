import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("avg_wait", "Average wait (s)", True),
    ("p95_wait", "P95 wait (s)", True),
    ("throughput", "Throughput (veh/min)", False),
    ("avg_queue", "Average queue", True),
    ("fairness", "Fairness", False),
]


def load_overall_mean(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if "iid" in df.columns:
        df = df[df["iid"].astype(str) == "-1"].copy()
    out = {}
    for metric, _, _ in METRICS:
        if metric in df.columns:
            out[metric] = pd.to_numeric(df[metric], errors="coerce").mean()
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot RL vs Neural performance across training time budgets")
    parser.add_argument("--neural-1m", required=True)
    parser.add_argument("--neural-3m", required=True)
    parser.add_argument("--neural-5m", required=True)
    parser.add_argument("--neural-10m", required=True)
    parser.add_argument("--rl-1m", required=True)
    parser.add_argument("--rl-3m", required=True)
    parser.add_argument("--rl-5m", required=True)
    parser.add_argument("--rl-10m", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title-prefix", default="Training budget")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = ["1m", "3m", "5m", "10m"]

    neural_paths = {
        "1m": Path(args.neural_1m),
        "3m": Path(args.neural_3m),
        "5m": Path(args.neural_5m),
        "10m": Path(args.neural_10m),
    }
    rl_paths = {
        "1m": Path(args.rl_1m),
        "3m": Path(args.rl_3m),
        "5m": Path(args.rl_5m),
        "10m": Path(args.rl_10m),
    }

    neural_data = {b: load_overall_mean(neural_paths[b]) for b in budgets}
    rl_data = {b: load_overall_mean(rl_paths[b]) for b in budgets}

    # Save merged table
    rows = []
    for b in budgets:
        row = {"budget": b}
        for metric, _, _ in METRICS:
            row[f"neural_{metric}"] = neural_data[b].get(metric, np.nan)
            row[f"rl_{metric}"] = rl_data[b].get(metric, np.nan)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "training_budget_summary.csv", index=False)

    x = np.arange(len(budgets))
    width = 0.35

    for metric, ylabel, lower_is_better in METRICS:
        neural_vals = [neural_data[b].get(metric, np.nan) for b in budgets]
        rl_vals = [rl_data[b].get(metric, np.nan) for b in budgets]

        plt.figure(figsize=(8, 5))
        plt.bar(x - width / 2, neural_vals, width, label="neural")
        plt.bar(x + width / 2, rl_vals, width, label="rl")
        plt.xticks(x, budgets)
        plt.xlabel("Training time budget")
        plt.ylabel(ylabel)
        plt.title(f"{args.title_prefix} {metric} comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"budget_{metric}_bar.png", dpi=200)
        plt.close()

    # Optional combined plot for avg_wait, often the most presentation-friendly
    metric = "avg_wait"
    neural_vals = [neural_data[b].get(metric, np.nan) for b in budgets]
    rl_vals = [rl_data[b].get(metric, np.nan) for b in budgets]
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, neural_vals, width, label="neural")
    plt.bar(x + width / 2, rl_vals, width, label="rl")
    plt.xticks(x, budgets)
    plt.xlabel("Training time budget")
    plt.ylabel("Average wait (s)")
    plt.title(f"{args.title_prefix} average wait")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "budget_main_avg_wait.png", dpi=200)
    plt.close()

    print(f"Saved training-budget charts to: {out_dir}")


if __name__ == "__main__":
    main()
