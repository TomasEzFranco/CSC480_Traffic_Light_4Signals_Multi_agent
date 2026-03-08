# Traffic Simulation Experiment Runbook

This document covers:
- Running single simulations
- Running batch experiments
- Training RL agents and saving weights
- Evaluating RL models
- Auto-selecting the best RL model
- Comparing best RL vs baseline controllers
- Generating plots

## 1. Go To Project Folder

```bash
cd /Users/quasar/Documents/CSC480/TrafficLightFinalProject/CSC480traffic_project_updated_RL_4_Intersections
```

## 2. Single Run (Quick Check)

Headless run with one controller:

```bash
python3 simulation.py \
  --headless \
  --mode adaptive \
  --seed 42 \
  --spawn-rate 1.0 \
  --duration 120 \
  --dt 0.0166667 \
  --experiment-id quick_check \
  --run-id adaptive_seed42_spawn1p00
```

Outputs:
- `results/quick_check/adaptive_seed42_spawn1p00/config.json`
- `results/quick_check/adaptive_seed42_spawn1p00/summary.json`
- `results/quick_check/adaptive_seed42_spawn1p00/intersection_timeseries.csv`

## 3. Batch Baseline Experiments

Run fixed/greedy/adaptive/random over multiple seeds/spawn rates:

```bash
python3 run_experiments.py \
  --modes fixed,greedy,adaptive,random \
  --seeds 42,43,44 \
  --spawn-rates 1.0,2.0 \
  --duration 180 \
  --dt 0.0166667 \
  --experiment-id baseline_v1
```

Output summary table:
- `results/baseline_v1/summary.csv`

## 4. Train RL Models

Train RL and save one model per run using a templated path:

```bash
python3 run_experiments.py \
  --modes rl \
  --seeds 42,43,44 \
  --spawn-rates 1.0,2.0 \
  --duration 180 \
  --dt 0.0166667 \
  --experiment-id rl_train_v2 \
  --rl-train \
  --rl-model-path "models/rl_{seed}_{run_id}.json"
```

Notes:
- `--rl-train` enables Q-learning updates.
- RL weights are saved at the end of each run.

## 5. Evaluate RL Models (No Training)

Use the same model-path template so each eval run loads its corresponding trained model:

```bash
python3 run_experiments.py \
  --modes rl \
  --seeds 42,43,44 \
  --spawn-rates 1.0,2.0 \
  --duration 180 \
  --dt 0.0166667 \
  --experiment-id rl_eval_v2 \
  --rl-model-path "models/rl_{seed}_{run_id}.json"
```

The RL model path used by each run is written into `summary.csv` as `rl_model_path`.

## 6. Auto-Select Best RL + Compare vs Baselines

One command can now:
1. Select the best RL model from a prior eval experiment (`rl_eval_v2`),
2. Copy it to `models/rl_best.json`,
3. Run the full baseline + RL comparison.

```bash
python3 run_experiments.py \
  --modes fixed,greedy,adaptive,random,rl \
  --seeds 42,43,44 \
  --spawn-rates 1.0,2.0 \
  --duration 180 \
  --dt 0.0166667 \
  --experiment-id compare_best_rl_v2 \
  --select-best-rl-from rl_eval_v2 \
  --select-best-rl-output models/rl_best.json
```

Additional artifact:
- `results/compare_best_rl_v2/best_rl_selection.json`

Best-model criterion used by the script:
- only rows where `iid = -1` and `mode = rl`
- lowest `avg_wait`
- tie-breakers: lower `red_light_violations`, then higher `throughput`

## 7. Generate Graphs

```bash
python3 visualize_results.py --experiment-id compare_best_rl_v2
```

Generated plots:
- `results/compare_best_rl_v2/mode_comparison.png`
- `results/compare_best_rl_v2/seed_boxplots.png`
- `results/compare_best_rl_v2/queue_timeseries.png` (if timeseries data available)

## 8. Confirm Selected Best Model

```bash
cat results/compare_best_rl_v2/best_rl_selection.json
```

## 9. Recommended End-to-End Command Sequence

```bash
cd /Users/quasar/Documents/CSC480/TrafficLightFinalProject/CSC480traffic_project_updated_RL_4_Intersections

python3 run_experiments.py \
  --modes rl \
  --seeds 42,43,44 \
  --spawn-rates 1.0,2.0 \
  --duration 180 \
  --dt 0.0166667 \
  --experiment-id rl_train_v2 \
  --rl-train \
  --rl-model-path "models/rl_{seed}_{run_id}.json"

python3 run_experiments.py \
  --modes rl \
  --seeds 42,43,44 \
  --spawn-rates 1.0,2.0 \
  --duration 180 \
  --dt 0.0166667 \
  --experiment-id rl_eval_v2 \
  --rl-model-path "models/rl_{seed}_{run_id}.json"

python3 run_experiments.py \
  --modes fixed,greedy,adaptive,random,rl \
  --seeds 42,43,44 \
  --spawn-rates 1.0,2.0 \
  --duration 180 \
  --dt 0.0166667 \
  --experiment-id compare_best_rl_v2 \
  --select-best-rl-from rl_eval_v2 \
  --select-best-rl-output models/rl_best.json

python3 visualize_results.py --experiment-id compare_best_rl_v2
```

## 10. Important Output Files Per Experiment

Inside `results/<experiment-id>/`:
- `summary.csv`: flattened metrics across runs and intersections (`iid = -1` is overall row)
- `<run-id>/config.json`: run configuration
- `<run-id>/summary.json`: nested summary for that run
- `<run-id>/intersection_timeseries.csv`: time-series metrics

## 11. Reproducibility Tips

- Keep `--dt`, `--duration`, `--seeds`, and `--spawn-rates` fixed across comparisons.
- Use a new `--experiment-id` for each study to avoid mixing outputs.
- Use `--no-log` during quick smoke tests if you do not need timeseries CSVs.

## 12. Train RL Longer (Recommended)

Because RL updates happen at phase-switch decisions, longer runs and repeated epochs improve learning.

```bash
cd /Users/quasar/Documents/CSC480/TrafficLightFinalProject/CSC480traffic_project_updated_RL_4_Intersections

for epoch in $(seq 1 20); do
  python3 run_experiments.py \
    --modes rl \
    --seeds 42,43,44 \
    --spawn-rates 1.0,2.0 \
    --duration 600 \
    --dt 0.0166667 \
    --experiment-id rl_train_v3_e${epoch} \
    --rl-train \
    --rl-model-path "models/rl_{seed}_{run_id}.json"
done
```

Then evaluate and compare:

```bash
python3 run_experiments.py \
  --modes rl \
  --seeds 42,43,44 \
  --spawn-rates 1.0,2.0 \
  --duration 180 \
  --dt 0.0166667 \
  --experiment-id rl_eval_v3 \
  --rl-model-path "models/rl_{seed}_{run_id}.json"

python3 run_experiments.py \
  --modes fixed,greedy,adaptive,random,rl \
  --seeds 42,43,44 \
  --spawn-rates 1.0,2.0 \
  --duration 180 \
  --dt 0.0166667 \
  --experiment-id compare_best_rl_v3 \
  --select-best-rl-from rl_eval_v3 \
  --select-best-rl-output models/rl_best.json

python3 visualize_results.py --experiment-id compare_best_rl_v3
```

## 13. RL Hyperparameter Sweep (Optional)

```bash
for alpha in 0.10 0.20 0.30; do
  python3 simulation.py \
    --headless \
    --mode rl \
    --seed 42 \
    --spawn-rate 1.0 \
    --duration 900 \
    --dt 0.0166667 \
    --experiment-id rl_hp_sweep \
    --run-id rl_a${alpha}_seed42_spawn1p00 \
    --rl-train \
    --rl-model-path "models/rl_a${alpha}_seed42_spawn1p00.json" \
    --rl-alpha ${alpha} \
    --rl-gamma 0.95 \
    --rl-epsilon 0.30 \
    --rl-epsilon-min 0.02 \
    --rl-epsilon-decay 0.9997
done
```

You can also pass RL hyperparameters through batch training now:

```bash
python3 run_experiments.py \
  --modes rl \
  --seeds 42,43,44 \
  --spawn-rates 1.0,2.0 \
  --duration 600 \
  --dt 0.0166667 \
  --experiment-id rl_train_hp_v1 \
  --rl-train \
  --rl-model-path "models/rlhp_{seed}_{run_id}.json" \
  --rl-alpha 0.15 \
  --rl-gamma 0.97 \
  --rl-epsilon 0.30 \
  --rl-epsilon-min 0.02 \
  --rl-epsilon-decay 0.9997
```

## 14. Epoch Study + RL-Only Comparison Graphs

`rl_epoch_study.py` automates:
1. train for `N` epochs,
2. snapshot models each epoch,
3. eval each epoch,
4. plot epoch metrics and per-model trend lines,
5. auto-pick the best model.

```bash
python3 rl_epoch_study.py \
  --study-id rl_epoch_study_v4 \
  --epochs 20 \
  --seeds 42,43,44 \
  --spawn-rates 1.0,2.0 \
  --duration-train 600 \
  --duration-eval 180 \
  --dt 0.0166667 \
  --rl-alpha 0.15 \
  --rl-gamma 0.97 \
  --rl-epsilon 0.30 \
  --rl-epsilon-min 0.02 \
  --rl-epsilon-decay 0.9997
```

Outputs:
- `results/rl_epoch_study_v4/epoch_summary.csv`
- `results/rl_epoch_study_v4/all_eval_rows.csv`
- `results/rl_epoch_study_v4/rl_epoch_metrics.png`
- `results/rl_epoch_study_v4/rl_epoch_run_comparison.png`
- `results/rl_epoch_study_v4/best_rl_selection.json`
- `models/rl_best_from_epoch_study.json`

## 15. RL Training Details (Current Implementation)

- Controller scope: one RL controller per intersection (`iid` 0..3), sharing one global Q-table.
- State per decision: `(current_phase, bin(q_right), bin(q_down), bin(q_left), bin(q_up))`.
- Queue binning: each direction bucketed by `q // 3`, capped at 8.
- Action: next phase index in `{0,1,2,3}`.
- Decision timing: RL updates on phase-switch decisions, not per frame.
- Reward: `(previous_total_queue - current_total_queue) - 0.05 * current_total_queue`.
- Policy: epsilon-greedy during training; greedy/tie-random at eval.
- Saved model: JSON Q-table + epsilon + metadata at `--rl-model-path`.
