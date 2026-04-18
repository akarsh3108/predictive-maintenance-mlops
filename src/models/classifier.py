"""PyTorch model architectures for predictive maintenance.

We use an MLP because the input is tabular sensor readings. An LSTM variant
(for sequential telemetry) is stubbed at the bottom for future work.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FailurePredictor(nn.Module):
    """Multi-layer perceptron binary classifier.

    Dropout and batch norm are used for regularization since the positive
    class is rare (~3.5%) and the model would otherwise overfit quickly.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = n_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # logit output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch,) logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return failure probability — used by the serving layer."""
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)
