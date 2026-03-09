import argparse
import csv
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from neural.model import NeuralPhasePolicy, save_model


FEATURE_COLUMNS = [
    "phase0", "phase1", "phase2", "phase3",
    "q_right", "q_down", "q_left", "q_up",
    "avg_wait_right", "avg_wait_down", "avg_wait_left", "avg_wait_up",
    "max_wait_right", "max_wait_down", "max_wait_left", "max_wait_up",
    "downstream_right", "downstream_down", "downstream_left", "downstream_up",
]


def load_csv(path):
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feats = [float(row[col]) for col in FEATURE_COLUMNS]
            label = int(row["action"])
            X.append(feats)
            y.append(label)
    return X, y


def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def iterate_minibatches(X, y, batch_size, shuffle=True):
    n = X.shape[0]
    idx = torch.arange(n)
    if shuffle:
        idx = idx[torch.randperm(n)]
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def evaluate(model, X, y, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = loss_fn(logits, y)
        acc = accuracy(logits, y)
    return float(loss.item()), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--time-budget-min", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-epochs", type=int, default=10000)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    X_train, y_train = load_csv(args.train_csv)
    X_val, y_val = load_csv(args.val_csv)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    counts = torch.bincount(y_train, minlength=4).float()
    class_weights = counts.sum() / torch.clamp(counts, min=1.0)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    model = NeuralPhasePolicy()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_state = None
    best_epoch = 0

    budget_seconds = float(args.time_budget_min) * 60.0
    start_time = time.time()
    deadline = start_time + budget_seconds
    epoch = 0

    print(f"time budget requested: {args.time_budget_min:.2f} min = {budget_seconds:.2f} sec")
    print(f"train class counts: {counts.tolist()}")

    while epoch < args.max_epochs and time.time() < deadline:
        epoch += 1
        model.train()

        for xb, yb in iterate_minibatches(X_train, y_train, args.batch_size, shuffle=True):
            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if time.time() >= deadline:
                break

        train_loss, train_acc = evaluate(model, X_train, y_train, loss_fn)
        val_loss, val_acc = evaluate(model, X_val, y_val, loss_fn)
        elapsed = time.time() - start_time

        print(
            f"epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"elapsed={elapsed:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training ended before any epoch completed.")

    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    save_model(model, args.save_path)

    elapsed = time.time() - start_time
    print(f"saved: {args.save_path}")
    print(f"best_val_acc: {best_val_acc}")
    print(f"best_epoch: {best_epoch}")
    print(f"elapsed_seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()