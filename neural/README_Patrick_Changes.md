# README: RL Protocol Fixes + Neural Baseline Pipeline

## Overview

This branch adds two major improvements to the project:

1. **RL evaluation protocol fixes**

   * uses a **shared RL model** across multiple training seeds
   * separates **train / validation / test seeds**
   * selects the best RL epoch using **average validation performance per epoch**
   * makes RL evaluation much more fair and reproducible

2. **Supervised neural baseline**

   * adds a new `neural` mode to the simulator
   * collects imitation-learning data from the `greedy` controller
   * trains a small MLP policy to imitate greedy decisions
   * allows comparison between:

     * `fixed`
     * `adaptive`
     * `greedy`
     * `random`
     * `rl`
     * `neural`

---

## Files changed

### Modified

* `simulation.py`
* `run_experiments.py`
* `rl_epoch_study.py`

### Added

* `collect_neural_data.py`
* `train_neural_policy.py`
* `neural/__init__.py`
* `neural/model.py`
* `neural/utils.py`

---

## What changed in RL

### `rl_epoch_study.py`

This script was updated so that RL evaluation is no longer done with overlapping seeds or one model per seed.

It now:

* uses separate seed splits:

  * **train seeds**
  * **validation seeds**
  * **test seeds**
* trains **one shared RL model**
* saves epoch snapshots
* picks the best epoch based on **mean validation performance**
* compares the final best model on held-out test seeds

This makes the RL comparison much more trustworthy.

---

## What changed in Neural

### `simulation.py`

Added:

* `--mode neural`
* `--neural-model-path`
* `--collect-neural-data`
* `--neural-data-path`

Also added:

* `NeuralController`
* neural model loading
* optional greedy teacher data collection during simulation

### `collect_neural_data.py`

Runs greedy in headless mode and collects state/action pairs into CSV.

### `train_neural_policy.py`

Trains a supervised MLP from collected greedy teacher data.

### `neural/model.py`

Defines the neural network.

### `neural/utils.py`

Encodes simulator state into model features.

---

# Setup

## 1. Clone the repo

```bash
git clone https://github.com/TomasEzFranco/CSC480_Traffic_Light_4Signals_Multi_agent.git
cd CSC480_Traffic_Light_4Signals_Multi_agent
```

---

## 2. Create a virtual environment

### Windows PowerShell

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install pygame matplotlib numpy torch
```

---

# Running the simulator

## Basic smoke test

### Headless greedy test

```bash
python simulation.py --headless --mode greedy --seed 42 --spawn-rate 1.0 --duration 120 --dt 0.0166667
```

### Headless neural test

```bash
python simulation.py --headless --mode neural --seed 42 --spawn-rate 1.0 --duration 120 --dt 0.0166667 --neural-model-path models/neural_greedy.pt
```

### Headless RL test

```bash
python simulation.py --headless --mode rl --seed 42 --spawn-rate 1.0 --duration 120 --dt 0.0166667 --rl-model-path models/rl_best_shared.json
```

---

# RL workflow

## Train a shared RL model

Example:

```bash
python run_experiments.py --modes rl --seeds 21,22,23,24,25,26,27,28,29,30 --spawn-rates 1.0 --duration 300 --dt 0.0166667 --experiment-id rl_train_shared_v1 --rl-train --rl-model-path models/rl_shared_v1.json --rl-alpha 0.15 --rl-gamma 0.97 --rl-epsilon 0.30 --rl-epsilon-min 0.02 --rl-epsilon-decay 0.9997
```

## Validate the trained RL model

```bash
python run_experiments.py --modes rl --seeds 31,32,33,34,35 --spawn-rates 1.0 --duration 180 --dt 0.0166667 --experiment-id rl_val_shared_v1 --rl-model-path models/rl_shared_v1.json
```

## Run epoch study

```bash
python rl_epoch_study.py --study-id rl_shared_protocol_test_v3 --epochs 3 --train-seeds 21,22,23,24,25,26,27,28,29,30 --val-seeds 31,32,33,34,35 --test-seeds 41,42,43,44,45 --spawn-rates 1.0 --duration-train 300 --duration-eval 180 --compare-duration 180 --model-path models/rl_shared_epoch.json --best-model-output models/rl_best_shared.json --run-compare
```

---

# Neural workflow

## 1. Collect training data from greedy teacher

```bash
python collect_neural_data.py --seeds 21,22,23,24,25,26,27,28,29,30 --spawn-rates 1.0 --duration 300 --data-path data/neural/greedy_train.csv
```

## 2. Collect validation data

```bash
python collect_neural_data.py --seeds 31,32,33,34,35 --spawn-rates 1.0 --duration 300 --data-path data/neural/greedy_val.csv
```

## 3. Train the neural model

```bash
python train_neural_policy.py --train-csv data/neural/greedy_train.csv --val-csv data/neural/greedy_val.csv --save-path models/neural_greedy.pt --epochs 30
```

## 4. Smoke test the neural controller

```bash
python simulation.py --headless --mode neural --seed 42 --spawn-rate 1.0 --duration 120 --dt 0.0166667 --neural-model-path models/neural_greedy.pt --experiment-id neural_smoke --run-id neural_seed42
```

---

# Full comparison on held-out test seeds

This runs all baselines plus RL and neural on the held-out test set:

```bash
python run_experiments.py --modes fixed,adaptive,greedy,random,rl,neural --seeds 41,42,43,44,45 --spawn-rates 1.0 --duration 180 --dt 0.0166667 --experiment-id compare_with_neural --neural-model-path models/neural_greedy.pt --rl-model-path models/rl_best_shared.json
```

Output summary will be written to:

```text
results/compare_with_neural/summary.csv
```

---

# Important notes

## Headless mode

For all experiments, use `--headless`.
This avoids GUI timing differences and makes results more reproducible.

## Artifacts

By default, large generated artifacts such as:

* `results/`
* `data/neural/*.csv`
* `models/*.pt`
* `models/*.json`

may or may not be committed depending on team preference.
If they are not in the repo, regenerate them locally using the commands above.

## Current performance status

At the moment:

* `greedy` is still the strongest baseline on held-out seeds
* `rl` is now evaluated under a much fairer protocol
* `neural` works and outperforms `random`, but still does not match `greedy`

---

# How to push your own changes to a new branch

## 1. Clone the repo

```bash
git clone https://github.com/TomasEzFranco/CSC480_Traffic_Light_4Signals_Multi_agent.git
cd CSC480_Traffic_Light_4Signals_Multi_agent
```

## 2. Create a new branch

```bash
git checkout -b your-branch-name
```

Example:

```bash
git checkout -b patrick-rl-protocol-neural
```

## 3. Make your code changes

## 4. Check git status

```bash
git status
```

## 5. Stage files

```bash
git add .
```

## 6. Commit

```bash
git commit -m "Describe your changes here"
```

Example:

```bash
git commit -m "Add RL protocol fixes and neural baseline pipeline"
```

## 7. Push to GitHub

```bash
git push -u origin your-branch-name
```

Example:

```bash
git push -u origin patrick-rl-protocol-neural
```

## 8. Open a Pull Request

After pushing, GitHub will give you a link to open a PR.

---

# Recommended branch naming

Examples:

* `patrick-rl-protocol-neural`
* `add-neural-baseline`
* `fix-rl-eval-protocol`

---
