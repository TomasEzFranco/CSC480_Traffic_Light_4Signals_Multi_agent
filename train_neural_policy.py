import argparse
import csv
import random

import torch
import torch.nn as nn
import torch.optim as optim

from neural.model import NeuralPhasePolicy, save_model


def load_csv(path):
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feats = [
                float(row["phase0"]),
                float(row["phase1"]),
                float(row["phase2"]),
                float(row["phase3"]),
                float(row["q_right"]),
                float(row["q_down"]),
                float(row["q_left"]),
                float(row["q_up"]),
            ]
            label = int(row["action"])
            X.append(feats)
            y.append(label)
    return X, y


def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description="Train supervised neural phase policy")
    parser.add_argument("--train-csv", default="data/neural/greedy_train.csv")
    parser.add_argument("--val-csv", default="data/neural/greedy_val.csv")
    parser.add_argument("--save-path", default="models/neural_greedy.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    opt = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            train_acc = accuracy(model(X_train), y_train)
            val_acc = accuracy(model(X_val), y_val)

        print(
            f"epoch {epoch+1:02d} "
            f"loss={loss.item():.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    save_model(model, args.save_path)
    print("saved:", args.save_path)
    print("best_val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
