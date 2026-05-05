from __future__ import annotations

import torch.nn as nn


class RiskClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)
