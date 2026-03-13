# Traffic Simulation Experiments (RL + Neural)

This runbook documents the current reproducible protocol for RL and Neural models.

## 1. Fixed Seed Split (Use This Everywhere)

- Train seeds: `40,41,42,43,44,45,46,47`
- Validation seeds: `48,49,50`
- Test seeds: `51,52,53`

Keep train/val/test disjoint for both RL and neural.

## 2. Go To Project Folder

```bash
cd /Users/quasar/Documents/CSC480/TrafficLightFinalProject/CSC480traffic_project_hybrid_fresh
```

## 3. Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

## 4. Why Tune Hyperparameters First (Short Justification)

We tune RL hyperparameters first because tabular Q-learning is sensitive to reward scaling and switch penalties.

- Better balance across throughput gain, wait reduction, and switch cost can reduce avg wait.
- A short sweep is much cheaper than running a full 40-epoch study repeatedly.
- It makes the final long run defensible: we pre-select one config on validation only, then lock it.

## 5. RL Parameter Study (Short Sweep)

Run a short sweep first:

```bash
python3 rl_reward_parametric_study.py \
  --study-id rl_param_sweep_40_53_v1 \
  --epochs 6 \
  --train-seeds 40,41,42,43,44,45,46,47 \
  --val-seeds 48,49,50 \
  --test-seeds 51,52,53 \
  --spawn-rates 1.0 \
  --duration-train 180 \
  --duration-eval 90 \
  --dt 0.0166667 \
  --rl-alpha 0.20 \
  --rl-gamma 0.95 \
  --rl-epsilon 0.20 \
  --rl-epsilon-min 0.02 \
  --rl-epsilon-decay 0.9995 \
  --rl-state-profile fine \
  --no-log
```

Extract these artifacts:

- `results/rl_param_sweep_40_53_v1/combined_epoch_metrics.csv`: every epoch for every config.
- `results/rl_param_sweep_40_53_v1/best_config_by_validation.csv`: best epoch per config using validation metrics.
- `results/rl_param_sweep_40_53_v1/reward_sweep_epoch_dashboard.png`: learning curves.
- `results/rl_param_sweep_40_53_v1/reward_sweep_best_tradeoff.png`: wait/throughput frontier across reward presets.

What to pick from the sweep:

- Choose one final reward config using validation-only metrics:
- Lowest `avg_wait_mean`
- Tie-breaker: lowest `red_light_violations_mean`
- Tie-breaker: highest `throughput_mean`
- Copy the selected values for `--rl-w-throughput`, `--rl-w-wait-delta`, and `--rl-switch-penalty`.

## 6. RL Full Epoch Study (Long Run for Best RL Model)

This long run does all RL stages:
1. trains one shared Q-table on train seeds each epoch,
2. validates snapshots on val seeds each epoch,
3. selects best epoch by validation means only,
4. copies best model,
5. runs holdout compare on test seeds.

```bash
python3 rl_epoch_study.py \
  --study-id rl_split_40_53_v1 \
  --epochs 40 \
  --train-seeds 40,41,42,43,44,45,46,47 \
  --val-seeds 48,49,50 \
  --test-seeds 51,52,53 \
  --spawn-rates 1.0 \
  --duration-train 900 \
  --duration-eval 180 \
  --compare-duration 180 \
  --dt 0.0166667 \
  --rl-w-throughput 1.00 \
  --rl-w-wait-delta 0.050 \
  --rl-switch-penalty 0.20 \
  --rl-state-profile fine \
  --model-path models/studies/rl_split_40_53_v1/shared_qtable.json \
  --best-model-output models/rl_best_split_40_53_v1.json \
  --run-compare
```

Key outputs:
- `results/rl_split_40_53_v1/epoch_summary.csv`
- `results/rl_split_40_53_v1/val_epoch_summary.csv`
- `results/rl_split_40_53_v1/train_epoch_summary.csv`
- `results/rl_split_40_53_v1/all_val_rows.csv`
- `results/rl_split_40_53_v1/all_train_rows.csv`
- `results/rl_split_40_53_v1/rl_epoch_metrics.png`
- `results/rl_split_40_53_v1/rl_train_val_dashboard.png`
- `results/rl_split_40_53_v1/rl_epoch_improvement.png`
- `results/rl_split_40_53_v1/rl_epoch_run_comparison.png`
- `results/rl_split_40_53_v1/rl_val_epoch_boxplots.png`
- `results/rl_split_40_53_v1/best_rl_selection.json`
- `models/rl_best_split_40_53_v1.json`
- `results/rl_split_40_53_v1_compare_best/summary.csv`

### RL state bucket profiles (new)

Tabular RL now supports:
- `--rl-state-profile coarse`
- `--rl-state-profile default`
- `--rl-state-profile fine`

State tuple is unchanged:
- `(cur_phase, dom_q, dom_wait, q_bucket, max_wait_bucket)`

Approx table sizes:
- `coarse`: `4*4*4*7*7 = 3,136` states
- `default`: `4*4*4*8*8 = 4,096` states
- `fine`: `4*4*4*12*12 = 9,216` states

Practical guidance:
- use `fine` for your longer studies when you want more sensitivity to queue/wait changes
- use `default` for quick smoke runs
- use `coarse` only when you need faster, rougher convergence

### Simpler one-command RL workflow

Use this wrapper when you want the same train/val/test protocol with far fewer flags:

```bash
python3 rl_simple_study.py \
  --study-id rl_simple_40_53_v1 \
  --epochs 20
```

Defaults used by `rl_simple_study.py`:
- seeds: train `40-47`, val `48-50`, test `51-53`
- spawn rate: `2.0`
- reward: `w_throughput=1.00`, `w_wait_delta=0.050`, `switch_penalty=0.20`
- state profile: `fine`
- durations: train `300s`, eval `180s`, compare `180s`
- runs holdout compare automatically (use `--skip-compare` to disable)

Concrete alpha schedule for an 8-epoch run:

```bash
python3 rl_simple_study.py \
  --study-id rl_simple_sched_8e \
  --epochs 8 \
  --rl-alpha-schedule "1-4:0.2,5-6:0.14,7+:0.10"
```

## 7. RL Holdout Compare Graphs

```bash
python3 visualize_results.py --experiment-id rl_split_40_53_v1_compare_best
```

Generated:
- `results/rl_split_40_53_v1_compare_best/mode_comparison.png`
- `results/rl_split_40_53_v1_compare_best/seed_boxplots.png`
- `results/rl_split_40_53_v1_compare_best/queue_timeseries.png`

## 8. Neural Data Collection (Train + Val Splits)

Collect supervised data from teacher policy separately for train and validation seeds.

Train dataset:
```bash
python3 collect_neural_data.py \
  --seeds 40,41,42,43,44,45,46,47 \
  --spawn-rates 1.0 \
  --duration 300 \
  --dt 0.0166667 \
  --teacher-policy hybrid \
  --data-path data/neural/hybrid_train_40_47.csv
```

Validation dataset:
```bash
python3 collect_neural_data.py \
  --seeds 48,49,50 \
  --spawn-rates 1.0 \
  --duration 300 \
  --dt 0.0166667 \
  --teacher-policy hybrid \
  --data-path data/neural/hybrid_val_48_50.csv
```

## 9. Neural Training

```bash
python3 train_neural_policy.py \
  --train-csv data/neural/hybrid_train_40_47.csv \
  --val-csv data/neural/hybrid_val_48_50.csv \
  --save-path models/neural_split_40_53_v1.pt \
  --epochs 80 \
  --batch-size 256 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --patience 12 \
  --seed 123
```

`train_neural_policy.py` selects the checkpoint with best validation accuracy internally.

## 10. Neural Holdout Test (Test Seeds Only)

```bash
python3 run_experiments.py \
  --modes neural \
  --seeds 51,52,53 \
  --spawn-rates 1.0 \
  --duration 180 \
  --dt 0.0166667 \
  --experiment-id neural_test_40_53_v1 \
  --neural-model-path models/neural_split_40_53_v1.pt
```

Output:
- `results/neural_test_40_53_v1/summary.csv`

## 11. Final Holdout Comparison (All Modes, Same Test Seeds)

Use selected RL + trained neural on identical unseen test seeds:

```bash
python3 run_experiments.py \
  --modes fixed,greedy,adaptive,random,rl,neural \
  --seeds 51,52,53 \
  --spawn-rates 1.0 \
  --duration 180 \
  --dt 0.0166667 \
  --experiment-id compare_all_test_40_53_v1 \
  --rl-model-path models/rl_best_split_40_53_v1.json \
  --neural-model-path models/neural_split_40_53_v1.pt
```

Then plot:
```bash
python3 visualize_results.py --experiment-id compare_all_test_40_53_v1
```

## 12. Quick GUI Run (Visual Check)

Greedy GUI:
```bash
python3 simulation.py \
  --mode greedy \
  --seed 51 \
  --spawn-rate 1.0 \
  --duration 90 \
  --dt 0.0166667 \
  --fixed-green-seconds 3 \
  --shared-green-seconds 3
```

RL GUI (best split model):
```bash
python3 simulation.py \
  --mode rl \
  --seed 51 \
  --spawn-rate 1.0 \
  --duration 90 \
  --dt 0.0166667 \
  --fixed-green-seconds 3 \
  --shared-green-seconds 3 \
  --rl-model-path models/rl_best_split_40_53_v1.json
```

## 13. Reproducibility Notes

- Keep `--dt`, `--duration`, `--spawn-rates`, and seed splits fixed between model comparisons.
- Do not use test seeds (`51,52,53`) for model selection or hyperparameter tuning.
- Use a fresh `--experiment-id` per run.
- Use `--no-log` for faster sweeps if time-series CSV is not needed.
