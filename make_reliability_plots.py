import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "avg_wait",
    "p95_wait",
    "throughput",
    "avg_queue",
    "fairness",
]


LOWER_IS_BETTER = {
    "avg_wait": True,
    "p95_wait": True,
    "throughput": False,
    "avg_queue": True,
    "fairness": False,
}


def load_overall_rows(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)

    required_cols = {"mode", "seed", "iid"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"summary.csv is missing required columns: {sorted(missing)}")

    # overall rows are iid == -1
    df = df[df["iid"].astype(str) == "-1"].copy()

    if df.empty:
        raise ValueError("No overall rows found. Expected rows where iid == -1.")

    # convert metrics to numeric if present
    for col in METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df = df.sort_values(["mode", "seed"]).reset_index(drop=True)
    return df


def build_reliability_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for mode, g in df.groupby("mode"):
        row = {"mode": mode, "num_seeds": len(g)}

        for metric in METRICS:
            if metric not in g.columns:
                continue

            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(vals) == 0:
                continue

            mean_val = vals.mean()
            std_val = vals.std(ddof=1) if len(vals) > 1 else 0.0
            min_val = vals.min()
            max_val = vals.max()
            cv_val = (std_val / mean_val) if mean_val != 0 else np.nan
            spread_val = max_val - min_val

            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val
            row[f"{metric}_min"] = min_val
            row[f"{metric}_max"] = max_val
            row[f"{metric}_cv"] = cv_val
            row[f"{metric}_spread"] = spread_val

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("mode").reset_index(drop=True)
    return out


def build_reliability_rankings(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric in METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        spread_col = f"{metric}_spread"
        cv_col = f"{metric}_cv"

        if mean_col not in summary_df.columns:
            continue

        tmp = summary_df[["mode", mean_col, std_col, spread_col, cv_col]].copy()

        # For "performance rank", use mean only
        perf_ascending = LOWER_IS_BETTER[metric]
        tmp = tmp.sort_values(mean_col, ascending=perf_ascending).reset_index(drop=True)
        tmp["performance_rank"] = np.arange(1, len(tmp) + 1)

        # For "reliability rank", smaller std is always better
        tmp = tmp.sort_values(std_col, ascending=True).reset_index(drop=True)
        tmp["reliability_rank"] = np.arange(1, len(tmp) + 1)

        # Restore performance ordering for readability
        tmp = tmp.sort_values("performance_rank").reset_index(drop=True)
        tmp["metric"] = metric

        rows.append(tmp[[
            "metric",
            "mode",
            mean_col,
            std_col,
            spread_col,
            cv_col,
            "performance_rank",
            "reliability_rank",
        ]])

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def save_csvs(df_overall: pd.DataFrame, summary_df: pd.DataFrame, rankings_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_overall.to_csv(out_dir / "overall_rows.csv", index=False)
    summary_df.to_csv(out_dir / "reliability_summary.csv", index=False)
    rankings_df.to_csv(out_dir / "reliability_rankings.csv", index=False)


def plot_metric_errorbar(summary_df: pd.DataFrame, metric: str, out_dir: Path, title_prefix: str) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in summary_df.columns or std_col not in summary_df.columns:
        return

    plot_df = summary_df[["mode", mean_col, std_col]].copy().sort_values("mode")
    x = np.arange(len(plot_df))
    y = plot_df[mean_col].to_numpy(dtype=float)
    err = plot_df[std_col].to_numpy(dtype=float)

    plt.figure(figsize=(8, 5))
    plt.bar(x, y, yerr=err, capsize=5)
    plt.xticks(x, plot_df["mode"], rotation=20)
    plt.ylabel(metric)
    plt.title(f"{title_prefix} {metric} mean ± std")
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_reliability_bar.png", dpi=200)
    plt.close()


def plot_metric_by_seed(df_overall: pd.DataFrame, metric: str, out_dir: Path, title_prefix: str) -> None:
    if metric not in df_overall.columns:
        return

    plt.figure(figsize=(8, 5))

    for mode, g in df_overall.groupby("mode"):
        g = g.sort_values("seed")
        plt.plot(
            g["seed"].to_numpy(dtype=float),
            g[metric].to_numpy(dtype=float),
            marker="o",
            label=mode,
        )

    plt.xlabel("seed")
    plt.ylabel(metric)
    plt.title(f"{title_prefix} {metric} by seed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_by_seed.png", dpi=200)
    plt.close()


def print_console_summary(summary_df: pd.DataFrame) -> None:
    cols = []
    for metric in METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in summary_df.columns and std_col in summary_df.columns:
            cols.extend([mean_col, std_col])

    if not cols:
        print("No usable metric columns found.")
        return

    print("\n=== Reliability summary ===")
    print(summary_df[["mode"] + cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():
    parser = argparse.ArgumentParser(description="Create reliability summaries and plots from summary.csv")
    parser.add_argument("--summary-csv", required=True, help="Path to summary.csv from run_experiments")
    parser.add_argument("--out-dir", required=True, help="Directory to save csv summaries and plots")
    parser.add_argument("--title-prefix", default="", help="Optional title prefix for plots")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)

    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    df_overall = load_overall_rows(summary_csv)
    summary_df = build_reliability_summary(df_overall)
    rankings_df = build_reliability_rankings(summary_df)

    save_csvs(df_overall, summary_df, rankings_df, out_dir)

    for metric in METRICS:
        plot_metric_errorbar(summary_df, metric, out_dir, args.title_prefix)
        plot_metric_by_seed(df_overall, metric, out_dir, args.title_prefix)

    print_console_summary(summary_df)
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()