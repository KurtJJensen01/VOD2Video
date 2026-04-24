"""Model definitions and factories for highlight detection."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

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
        return self.network(features)


class CNNLSTMAudioHighlightModel(nn.Module):
    """Frozen ResNet18 frame encoder + LSTM temporal encoder + audio MLP head."""

    def __init__(
        self,
        *,
        lstm_hidden_dim: int = 256,
        audio_feature_dim: int = 7,
        dropout: float = 0.5,
        unfreeze_backbone: bool = False,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        self.frame_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.unfreeze_backbone = bool(unfreeze_backbone)
        self.resnet_feature_dim = int(backbone.fc.in_features)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
        )

        self.set_backbone_trainable(self.unfreeze_backbone)
        if not self.unfreeze_backbone:
            self.frame_encoder.eval()

        self.temporal_encoder = nn.LSTM(
            input_size=self.resnet_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim + audio_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.frame_encoder.parameters():
            parameter.requires_grad = bool(trainable)

    def train(self, mode: bool = True) -> "CNNLSTMAudioHighlightModel":
        super().train(mode)
        if not self.unfreeze_backbone:
            self.frame_encoder.eval()
        return self

    def forward(self, frames: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B, T, C, H, W].")
        if audio_features.ndim != 2:
            raise ValueError("audio_features must have shape [B, 7].")

        batch_size, sequence_length, channels, height, width = frames.shape
        normalized = (frames - self.imagenet_mean) / self.imagenet_std
        flat_frames = normalized.reshape(batch_size * sequence_length, channels, height, width)

        if self.unfreeze_backbone:
            encoded = self.frame_encoder(flat_frames)
        else:
            with torch.no_grad():
                encoded = self.frame_encoder(flat_frames)
        frame_embeddings = encoded.flatten(1).reshape(batch_size, sequence_length, self.resnet_feature_dim)
        _, (hidden_state, _) = self.temporal_encoder(frame_embeddings)
        video_embedding = hidden_state[-1]
        combined = torch.cat([video_embedding, audio_features], dim=1)
        return self.classifier(combined)


def build_model(config: ModelConfig) -> nn.Module:
    if config.model_name == "mlp_baseline":
        return MLPBaselineModel(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    if config.model_name == "cnn_lstm_audio":
        return CNNLSTMAudioHighlightModel(
            lstm_hidden_dim=config.lstm_hidden_dim,
            audio_feature_dim=config.audio_feature_dim,
            dropout=config.dropout,
            unfreeze_backbone=config.unfreeze_backbone,
            pretrained_backbone=config.pretrained_backbone,
        )
    raise ValueError(f"Unsupported model_name '{config.model_name}'.")
