

## 1. Folder: `results/final_main_comparison_hybrid_v3`

This folder is the **main held-out comparison** under the normal traffic setting.

### What this experiment measures
This is the most important comparison for the project. It compares all controllers on the held-out test seeds under the standard traffic load.

Typical modes included are:
- `fixed`
- `adaptive`
- `greedy`
- `random`
- `rl`
- `neural`

### Main files inside
- `summary.csv`  
  The raw result table for the experiment. This is the main source file used to generate graphs.
- `plots/`  
  Graphs that focus on **overall performance**.
- `reliability/`  
  Graphs and CSV summaries that focus on **stability across seeds**.

---

## RL metadata note (new)

For RL runs, each run folder's `config.json` now records:

- `rl_state_profile`
- `rl_q_bucket_width`
- `rl_q_bucket_cap`
- `rl_max_wait_bucket_width`
- `rl_max_wait_bucket_cap`
- `rl_state_table_size`

Use these fields to confirm RL comparisons were done with the same bucket granularity.

---

## 2. Folder: `results/final_main_comparison_hybrid_v3/plots`

This folder contains the standard performance plots.

### `overall_rows.csv`
Contains the overall result rows only, usually the rows where `iid = -1`.
This means each row is the full-experiment summary for one mode and one seed, not a single intersection.

### `mode_means.csv`
Contains the mean value of each metric for each mode, averaged across the test seeds.
This is useful when you want a single average number per method.

### `mode_stds.csv`
Contains the standard deviation of each metric for each mode across the test seeds.
This helps show how consistent or inconsistent a method is.

### `ranking_avg_wait.csv`
Ranks methods by average wait. Lower is better.

### `ranking_p95_wait.csv`
Ranks methods by p95 wait. Lower is better.
This is very useful for discussing tail behavior and whether some cars get stuck for too long.

### `ranking_throughput.csv`
Ranks methods by throughput. Higher is better.
This shows which method moves the most vehicles through the system.

### `ranking_avg_queue.csv`
Ranks methods by average queue. Lower is better.
This helps show congestion level.

### `ranking_fairness.csv`
Ranks methods by fairness. Higher is better.
This helps show whether the controller distributes service more evenly across directions.

### `avg_wait_bar.png`
Bar chart of **average wait time** by mode.
- Lower bars are better.
- This is usually the easiest graph to explain to an audience.
- Good for the question: **Which method makes cars wait less on average?**

### `p95_wait_bar.png`
Bar chart of **p95 wait time** by mode.
- Lower bars are better.
- This shows the tail of the wait distribution.
- Good for the question: **Which method avoids very bad worst-case waits for the slowest group of vehicles?**

### `throughput_bar.png`
Bar chart of **throughput** by mode.
- Higher bars are better.
- Good for the question: **Which method moves the most cars through the network?**

### `avg_queue_bar.png`
Bar chart of **average queue size** by mode.
- Lower bars are better.
- Good for the question: **Which method keeps congestion lower?**

### `fairness_bar.png`
Bar chart of **fairness** by mode.
- Higher bars are better.
- Good for the question: **Which method is less likely to neglect certain directions?**

### `avg_wait_by_seed.png`
Line plot of average wait across seeds.
- Each line is one mode.
- Good for showing whether results are stable or if a method is only good on certain seeds.

### `p95_wait_by_seed.png`
Line plot of p95 wait across seeds.
- Very useful for reliability discussions.
- Good for checking whether a method sometimes has a bad spike.

### `throughput_by_seed.png`
Line plot of throughput across seeds.
- Good for checking whether a controller is consistently efficient.

### `avg_queue_by_seed.png`
Line plot of average queue across seeds.
- Good for checking whether congestion changes a lot depending on seed.

---

## 3. Folder: `results/final_main_comparison_hybrid_v3/reliability`

This folder focuses on **reliability**, meaning how stable each method is across different seeds.

### Why this folder matters
A method may have a good average result but still be unreliable if it performs very differently depending on the seed.
Reliability analysis helps answer questions like:
- Does the method work consistently?
- Does it spike badly on some seeds?
- Is its tail behavior more stable?

### `reliability_summary.csv`
This CSV summarizes, for each mode and metric:
- mean
- standard deviation
- coefficient of variation
- min
- max
- range

How to interpret it:
- Lower `std` means the method is more stable.
- Lower `range` means the method changes less from seed to seed.
- Lower `cv` means the variation is small relative to the mean.

### `reliability_rankings.csv`
This CSV ranks methods by reliability-related statistics.
It helps quickly see which methods are most stable for each metric.

### `avg_wait_reliability_bar.png`
Bar chart of **mean average wait with error bars showing standard deviation**.
- Lower mean is better.
- Smaller error bars mean better reliability.

### `p95_wait_reliability_bar.png`
Bar chart of **mean p95 wait with standard deviation error bars**.
- Lower mean is better.
- Smaller error bars mean the tail behavior is more stable.
- This is especially useful if you want to argue that a method is more reliable even if its mean is not the absolute best.

### `throughput_reliability_bar.png`
Bar chart of **mean throughput with standard deviation error bars**.
- Higher mean is better.
- Smaller error bars mean more consistent throughput.

### `avg_queue_reliability_bar.png`
Bar chart of **mean average queue with standard deviation error bars**.
- Lower mean is better.
- Smaller error bars mean more stable congestion control.

### `fairness_reliability_bar.png`
Bar chart of **mean fairness with standard deviation error bars**.
- Higher mean is better.
- Smaller error bars mean the fairness behavior is more consistent.

### `avg_wait_by_seed.png`
Line plot of average wait by seed.
- Good for visually checking stability.
- If one line is flat and another line jumps around, the flatter one is more reliable.

### `p95_wait_by_seed.png`
Line plot of p95 wait by seed.
- Very important for tail reliability.
- Good for supporting claims like: **Hybrid neural may not have the absolute best mean wait, but it can be more stable in tail behavior.**

### `throughput_by_seed.png`
Line plot of throughput by seed.
- Good for showing whether efficiency is stable or volatile.

### `avg_queue_by_seed.png`
Line plot of average queue by seed.
- Good for showing whether congestion control is stable.

### `fairness_by_seed.png`
Line plot of fairness by seed.
- Good for showing whether a method consistently treats directions fairly.

---

## 4. Folder: `results/stress_compare_hybrid_v3`

This folder is the **stress test comparison**.

### What this experiment measures
This experiment uses a higher traffic load than the normal comparison.
It is used to test how controllers behave when traffic is harder and congestion pressure is stronger.

This folder has the same structure as the normal comparison folder:
- `summary.csv`
- `plots/`
- `reliability/`

The difference is that the stress experiment is meant to answer:
- Which method still works well under heavier load?
- Which method degrades gracefully?
- Which method becomes unstable when traffic gets harder?

---

## 5. Folder: `results/stress_compare_hybrid_v3/plots`

The graph names here mean the same thing as in:
- `results/final_main_comparison_hybrid_v3/plots`

The only difference is the interpretation:
these graphs tell you how each method performs **under higher traffic pressure**.

Important graphs to look at here:

### `avg_wait_bar.png`
Used to see which controller keeps wait times lower under stress.

### `p95_wait_bar.png`
Used to see which controller avoids very bad delays when traffic is heavy.
This is often one of the most important stress-test graphs.

### `throughput_bar.png`
Used to check whether the controller still moves a high number of vehicles through the network.

### `avg_queue_bar.png`
Used to check whether congestion builds up badly under load.

### `*_by_seed.png`
Used to see whether the stress performance is consistent or if some seeds make the controller collapse.

---

## 6. Folder: `results/stress_compare_hybrid_v3/reliability`

The graph names here also mean the same thing as in:
- `results/final_main_comparison_hybrid_v3/reliability`

But here the focus is specifically:

### Stress reliability
This folder helps answer:
- Which controller stays stable when traffic is harder?
- Which controller has smaller seed-to-seed variation under congestion?
- Which controller has better worst-case or tail behavior under stress?

In practice, this folder is very useful if you want to argue that a method is not just good on average, but also **robust**.

---

## 7. Quick guide

If you only want the most important files, start here:

### For normal-load main performance
- `results/final_main_comparison_hybrid_v3/plots/avg_wait_bar.png`
- `results/final_main_comparison_hybrid_v3/plots/p95_wait_bar.png`
- `results/final_main_comparison_hybrid_v3/plots/throughput_bar.png`

### For normal-load reliability
- `results/final_main_comparison_hybrid_v3/reliability/reliability_summary.csv`
- `results/final_main_comparison_hybrid_v3/reliability/p95_wait_reliability_bar.png`
- `results/final_main_comparison_hybrid_v3/reliability/p95_wait_by_seed.png`

### For stress-test performance
- `results/stress_compare_hybrid_v3/plots/avg_wait_bar.png`
- `results/stress_compare_hybrid_v3/plots/p95_wait_bar.png`
- `results/stress_compare_hybrid_v3/plots/throughput_bar.png`

### For stress-test reliability
- `results/stress_compare_hybrid_v3/reliability/reliability_summary.csv`
- `results/stress_compare_hybrid_v3/reliability/p95_wait_reliability_bar.png`
- `results/stress_compare_hybrid_v3/reliability/p95_wait_by_seed.png`

---

## 8. Recommended presentation use

### Best graphs for a simple story
- `avg_wait_bar.png`
- `p95_wait_bar.png`
- `throughput_bar.png`

### Best graphs for a reliability story
- `p95_wait_reliability_bar.png`
- `p95_wait_by_seed.png`
- `reliability_summary.csv`

### Best graphs for a stress-test story
- `results/stress_compare_hybrid_v3/plots/avg_wait_bar.png`
- `results/stress_compare_hybrid_v3/plots/p95_wait_bar.png`
- `results/stress_compare_hybrid_v3/reliability/p95_wait_by_seed.png`

