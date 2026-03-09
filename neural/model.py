import os
import torch
import torch.nn as nn

INPUT_DIM = 20
NUM_CLASSES = 4


class NeuralPhasePolicy(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(), 
            nn.Linear(160, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def save_model(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path, device="cpu"):
    model = NeuralPhasePolicy()
    state = torch.load(path, map_location=device)
    # Backward compatibility:
    # older checkpoints saved a bare nn.Sequential and use keys like "0.weight"
    # while current model wraps it as self.net with keys like "net.0.weight".
    if isinstance(state, dict):
        has_prefixed = any(k.startswith("net.") for k in state.keys())
        has_unprefixed = any(
            k.startswith(("0.", "1.", "2.", "3.", "4.", "5."))
            for k in state.keys()
        )
        if has_unprefixed and not has_prefixed:
            state = {f"net.{k}": v for k, v in state.items()}
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Incompatible neural checkpoint at {path}. "
            "This build expects the hybrid policy architecture "
            "(20 input features, 4 output phases). "
            "Use models/neural_hybrid.pt or retrain with train_neural_policy.py."
        ) from exc
    model.eval()
    return model
