import argparse
import csv
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
            feats = [float(row[c]) for c in FEATURE_COLUMNS]
            label = int(row["action"])
            X.append(feats)
            y.append(label)
    return X, y


def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def evaluate(model, X, y, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = loss_fn(logits, y).item()
        acc = accuracy(logits, y)
    return loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/neural/hybrid_train.csv")
    parser.add_argument("--val-csv", default="data/neural/hybrid_val.csv")
    parser.add_argument("--save-path", default="models/neural_hybrid.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
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
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    model = NeuralPhasePolicy()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_val_acc = -1.0
    best_state = None
    epochs_without_improve = 0

    print("train class counts:", counts.tolist())

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_sum += loss.item() * xb.size(0)
            train_correct += (logits.argmax(dim=1) == yb).sum().item()
            train_total += xb.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_loss, val_acc = evaluate(model, X_val, y_val, loss_fn)

        print(
            f"epoch {epoch+1:02d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.patience:
            print(f"early stopping after epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    save_model(model, args.save_path)
    print("saved:", args.save_path)
    print("best_val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
