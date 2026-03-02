import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim

DATA = "supervised_data.csv"
SAVE_AS = "model_supervised.pt"
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)

# -------- Load data --------
rows = []
with open(DATA, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        if len(row) != 7:
            continue
        rows.append(row)

if len(rows) < 100:
    raise ValueError(f"Not enough data in {DATA}. Got {len(rows)} rows. Collect more first.")

X_list, y_list = [], []
for row in rows:
    *feat, label = row
    X_list.append([float(v) for v in feat])
    y_list.append(int(label))

# Shuffle
idx = list(range(len(X_list)))
random.shuffle(idx)
X_list = [X_list[i] for i in idx]
y_list = [y_list[i] for i in idx]

# Train/val split
split = int(0.8 * len(X_list))
X_train, y_train = X_list[:split], y_list[:split]
X_val, y_val     = X_list[split:], y_list[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val   = torch.tensor(X_val, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.long)

# -------- Class-weighted loss (fix imbalance) --------
counts = torch.bincount(y_train)  # counts[0], counts[1]
w0 = 1.0
w1 = counts[0].item() / max(1, counts[1].item())
class_w = torch.tensor([w0, w1], dtype=torch.float32)
loss_fn = nn.CrossEntropyLoss(weight=class_w)

print("class counts:", counts.tolist(), "weights:", class_w.tolist())

# -------- Model --------
model = nn.Sequential(
    nn.Linear(6, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 2)
)

opt = optim.Adam(model.parameters(), lr=1e-3)

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

# -------- Train --------
for epoch in range(30):
    model.train()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            train_acc = accuracy(model(X_train), y_train)
            val_acc = accuracy(model(X_val), y_val)
        print(f"epoch {epoch+1:02d} loss={loss.item():.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

torch.save(model.state_dict(), SAVE_AS)
print("saved", SAVE_AS)