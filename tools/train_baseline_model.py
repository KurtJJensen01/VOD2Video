#!/usr/bin/env python3
"""CLI entry point for the Phase 2B training framework."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.models import build_model  # noqa: E402
from vod2video.training import build_loss_function, set_random_seed, train_model, validate_model  # noqa: E402
from vod2video.training_config import (  # noqa: E402
    CheckpointConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
)
from vod2video.training_data import (  # noqa: E402
    DEFAULT_FEATURE_NAMES,
    TrainingDataError,
    build_dataloaders_from_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the baseline model from a split manifest or precomputed feature manifest."
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=REPO_ROOT / "artifacts" / "splits" / "branch_1c" / "all_splits.csv",
        help="Path to a split manifest CSV or a precomputed feature manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "training" / "branch_2b_baseline",
        help="Directory used for checkpoints and training history output.",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=16, help="Hidden layer width.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for the baseline MLP.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="f1",
        choices=("loss", "accuracy", "precision", "recall", "f1"),
        help="Validation metric used for best-model checkpointing.",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=list(DEFAULT_FEATURE_NAMES),
        help="Numeric feature columns to feed into the baseline model.",
    )
    return parser.parse_args()


def summarize_epoch(history: dict[str, object], epoch_index: int) -> str:
    epoch_record = history["epochs"][epoch_index]
    train_metrics = epoch_record["train"]
    val_metrics = epoch_record["val"]
    return (
        f"epoch {epoch_record['epoch']:02d} | "
        f"train loss={train_metrics['loss']:.4f} f1={train_metrics['f1']:.4f} | "
        f"val loss={val_metrics['loss']:.4f} f1={val_metrics['f1']:.4f} "
        f"acc={val_metrics['accuracy']:.4f}"
    )


def main() -> int:
    args = parse_args()

    try:
        data_config = DataConfig(
            split_manifest_path=args.split_manifest,
            batch_size=args.batch_size,
        )
        bundle = build_dataloaders_from_manifest(
            data_config,
            feature_names=args.feature_columns,
        )
    except (FileNotFoundError, TrainingDataError, ValueError) as exc:
        print(f"Training data setup failed: {exc}", file=sys.stderr)
        return 1

    model_config = ModelConfig(
        input_dim=bundle.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        random_seed=args.seed,
        device=args.device,
        checkpoint=CheckpointConfig(
            output_dir=args.output_dir,
            monitor_metric=args.monitor_metric,
            monitor_mode="min" if args.monitor_metric == "loss" else "max",
        ),
    )

    set_random_seed(training_config.random_seed)
    model = build_model(model_config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history = train_model(
        model=model,
        model_config=model_config,
        train_loader=bundle.dataloaders[data_config.train_split_name],
        val_loader=bundle.dataloaders[data_config.val_split_name],
        optimizer=optimizer,
        training_config=training_config,
        feature_names=list(bundle.feature_names),
        normalization_stats=bundle.normalization_stats,
    )

    model_for_test = build_model(model_config)
    best_checkpoint = torch.load(history["best_checkpoint_path"], map_location=training_config.device)
    model_for_test.load_state_dict(best_checkpoint["model_state_dict"])
    model_for_test = model_for_test.to(training_config.device)
    loss_fn = build_loss_function(training_config, device=torch.device(training_config.device))
    test_metrics = validate_model(
        model_for_test,
        bundle.dataloaders[data_config.test_split_name],
        device=torch.device(training_config.device),
        loss_fn=loss_fn,
        threshold=training_config.decision_threshold,
    )

    print("Training complete")
    print(
        f"manifest={args.split_manifest.resolve()} "
        f"features={', '.join(bundle.feature_names)} "
        f"device={training_config.device}"
    )
    print("normalization means=" + json.dumps(bundle.normalization_stats["means"]))
    print("normalization stds=" + json.dumps(bundle.normalization_stats["stds"]))
    for index in range(len(history["epochs"])):
        print(summarize_epoch(history, index))
    print()
    print(
        f"best epoch={history['best_epoch']} "
        f"monitor={training_config.checkpoint.monitor_metric} "
        f"value={history['best_metric_value']:.4f}"
    )
    print(f"best checkpoint={history['best_checkpoint_path']}")
    print(f"latest checkpoint={history['latest_checkpoint_path']}")
    print(f"history json={history['history_path']}")
    print("test metrics=" + json.dumps(test_metrics.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
