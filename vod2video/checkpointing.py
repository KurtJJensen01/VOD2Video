"""Checkpoint persistence helpers for training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .training_config import CheckpointConfig


def ensure_checkpoint_dir(config: CheckpointConfig) -> Path:
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def is_better_metric(candidate: float, current_best: float | None, mode: str) -> bool:
    if current_best is None:
        return True
    if mode == "max":
        return candidate > current_best
    if mode == "min":
        return candidate < current_best
    raise ValueError(f"Unsupported checkpoint monitor_mode: {mode}")


def build_checkpoint_payload(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    metrics: dict[str, float | int],
    feature_names: list[str],
    normalization_stats: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "training_config": training_config,
        "metrics": metrics,
        "feature_names": feature_names,
    }
    if normalization_stats is not None:
        payload["normalization_stats"] = normalization_stats
    return payload


def save_checkpoint(
    payload: dict[str, Any],
    *,
    checkpoint_path: Path,
) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def save_training_history(history: dict[str, Any], config: CheckpointConfig) -> Path:
    output_dir = ensure_checkpoint_dir(config)
    history_path = output_dir / config.history_filename
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history_path
