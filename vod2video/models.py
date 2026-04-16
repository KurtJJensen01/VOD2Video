"""Model definitions and factories for the Phase 2B training framework."""

from __future__ import annotations

import torch
from torch import nn

from .training_config import ModelConfig


class MLPBaselineModel(nn.Module):
    """Small baseline classifier for metadata placeholder features."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def build_model(config: ModelConfig) -> nn.Module:
    if config.model_name != "mlp_baseline":
        raise ValueError(
            f"Unsupported model_name '{config.model_name}'. "
            "Phase 2B currently ships with the 'mlp_baseline' placeholder."
        )
    return MLPBaselineModel(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )
