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
    model.load_state_dict(state)
    model.eval()
    return model
