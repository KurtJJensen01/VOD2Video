"""Reusable training and validation loops for Phase 2B."""

from __future__ import annotations

from dataclasses import asdict
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from .checkpointing import (
    build_checkpoint_payload,
    ensure_checkpoint_dir,
    is_better_metric,
    save_checkpoint,
    save_training_history,
)
from .metrics import BinaryClassificationMetrics, compute_binary_classification_metrics, sweep_thresholds
from .training_config import ModelConfig, TrainingConfig


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loss_function(training_config: TrainingConfig, device: torch.device) -> nn.Module:
    if training_config.positive_class_weight is None:
        return nn.BCEWithLogitsLoss()
    pos_weight = torch.tensor([training_config.positive_class_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _run_model_on_loader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    total_loss = 0.0
    total_examples = 0
    collected_logits: list[torch.Tensor] = []
    collected_labels: list[torch.Tensor] = []

    is_training = optimizer is not None
    model.train(mode=is_training)

    for batch in dataloader:
        labels = batch["label"].to(device).view(-1, 1)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = _forward_batch(model, batch, device=device)
            loss = loss_fn(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        collected_logits.append(logits.detach().cpu())
        collected_labels.append(labels.detach().cpu())

    if total_examples == 0:
        raise ValueError("Dataloader produced zero examples.")

    average_loss = total_loss / total_examples
    return average_loss, torch.cat(collected_logits), torch.cat(collected_labels)


def _forward_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    if "frames" in batch and "audio_features" in batch:
        frames = batch["frames"].to(device)
        audio_features = batch["audio_features"].to(device)
        return model(frames, audio_features)
    features = batch["features"].to(device)
    return model(features)


def validate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    average_loss, logits, labels = _run_model_on_loader(
        model,
        dataloader,
        device=device,
        loss_fn=loss_fn,
        optimizer=None,
    )
    return compute_binary_classification_metrics(
        logits,
        labels,
        loss=average_loss,
        threshold=threshold,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    average_loss, logits, labels = _run_model_on_loader(
        model,
        dataloader,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )
    return compute_binary_classification_metrics(
        logits,
        labels,
        loss=average_loss,
        threshold=threshold,
    )


def train_model(
    *,
    model: nn.Module,
    model_config: ModelConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    feature_names: list[str],
    normalization_stats: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    set_random_seed(training_config.random_seed)
    device = torch.device(training_config.device)
    checkpoint_dir = ensure_checkpoint_dir(training_config.checkpoint)

    model = model.to(device)
    loss_fn = build_loss_function(training_config, device=device)

    history: dict[str, Any] = {
        "epochs": [],
        "best_epoch": None,
        "best_metric_value": None,
        "monitor_metric": training_config.checkpoint.monitor_metric,
        "checkpoint_dir": str(checkpoint_dir),
    }

    best_metric_value: float | None = None
    best_checkpoint_path: str | None = None
    latest_checkpoint_path: str | None = None

    for epoch in range(1, training_config.epochs + 1):
        train_loss, train_logits, train_labels = _run_model_on_loader(
            model,
            train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        val_loss, val_logits, val_labels = _run_model_on_loader(
            model,
            val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=None,
        )
        chosen_threshold, val_metrics = sweep_thresholds(
            val_logits,
            val_labels,
            loss=val_loss,
            min_precision=0.15,
        )
        train_metrics = compute_binary_classification_metrics(
            train_logits,
            train_labels,
            loss=train_loss,
            threshold=chosen_threshold,
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics.to_dict(),
            "val": val_metrics.to_dict(),
            "chosen_threshold": chosen_threshold,
        }
        history["epochs"].append(epoch_record)

        monitor_metric_name = training_config.checkpoint.monitor_metric
        monitored_value = float(epoch_record["val"][monitor_metric_name])
        serializable_training_config = training_config.to_serializable_dict()
        serializable_training_config["decision_threshold"] = float(chosen_threshold)
        checkpoint_payload = build_checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            model_config=asdict(model_config),
            training_config=serializable_training_config,
            metrics=epoch_record["val"],
            feature_names=feature_names,
            normalization_stats=normalization_stats,
        )

        if training_config.checkpoint.save_latest:
            latest_path = checkpoint_dir / training_config.checkpoint.latest_filename
            latest_checkpoint_path = str(save_checkpoint(checkpoint_payload, checkpoint_path=latest_path))

        if is_better_metric(
            monitored_value,
            best_metric_value,
            training_config.checkpoint.monitor_mode,
        ):
            best_metric_value = monitored_value
            history["best_epoch"] = epoch
            history["best_metric_value"] = monitored_value
            best_path = checkpoint_dir / training_config.checkpoint.best_filename
            best_checkpoint_path = str(save_checkpoint(checkpoint_payload, checkpoint_path=best_path))

    history["best_checkpoint_path"] = best_checkpoint_path
    history["latest_checkpoint_path"] = latest_checkpoint_path
    history_path = save_training_history(history, training_config.checkpoint)
    history["history_path"] = str(history_path)
    return history
