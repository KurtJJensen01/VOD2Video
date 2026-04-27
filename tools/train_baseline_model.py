#!/usr/bin/env python3
"""CLI entry point for the Phase 2B training framework."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.models import build_model  # noqa: E402
from vod2video.metrics import sweep_thresholds  # noqa: E402
from vod2video.evaluation import evaluate_checkpoint_on_manifest  # noqa: E402
from vod2video.training import set_random_seed, train_model  # noqa: E402
from vod2video.training_config import (  # noqa: E402
    CheckpointConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
)
from vod2video.training_data import (  # noqa: E402
    DEFAULT_FEATURE_NAMES,
    TrainingDataError,
    build_video_audio_dataloaders_from_manifest,
    compute_positive_class_weight_from_manifest,
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
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Optimizer weight decay.")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop training after this many consecutive epochs without validation F1 improvement.",
    )
    parser.add_argument("--hidden-dim", type=int, default=16, help="Hidden layer width.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for the baseline MLP.")
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Unfreeze the ResNet18 backbone for experimentation.",
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        help="Explicit positive-class loss weight. By default this is derived from the train split.",
    )
    parser.add_argument(
        "--disable-auto-class-weight",
        action="store_true",
        help="Disable automatic positive-class weighting from the train split.",
    )
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
        f"acc={val_metrics['accuracy']:.4f} "
        f"threshold={epoch_record.get('chosen_threshold', 0.5):.2f}"
    )


def choose_threshold_from_val_predictions(predictions: pd.DataFrame) -> float:
    probabilities = torch.tensor(
        predictions["predicted_probability"].to_numpy(dtype="float32"),
        dtype=torch.float32,
    ).clamp(1e-6, 1.0 - 1e-6)
    labels = torch.tensor(predictions["label"].to_numpy(dtype="float32"), dtype=torch.float32)
    threshold, _ = sweep_thresholds(torch.logit(probabilities), labels, loss=float("nan"), min_precision=0.15)
    return float(threshold)


def main() -> int:
    args = parse_args()

    try:
        data_config = DataConfig(
            split_manifest_path=args.split_manifest,
            batch_size=args.batch_size,
        )
        bundle = build_video_audio_dataloaders_from_manifest(data_config)
        if args.positive_class_weight is not None:
            positive_class_weight = float(args.positive_class_weight)
        elif args.disable_auto_class_weight:
            positive_class_weight = None
        else:
            positive_class_weight = compute_positive_class_weight_from_manifest(
                args.split_manifest,
                split_column=data_config.split_column,
                label_column=data_config.label_column,
                train_split_name=data_config.train_split_name,
                cap=10.0,
            )
    except (FileNotFoundError, TrainingDataError, ValueError) as exc:
        print(f"Training data setup failed: {exc}", file=sys.stderr)
        return 1

    model_config = ModelConfig(
        model_name="cnn_lstm_audio",
        dropout=0.5,
        lstm_hidden_dim=256,
        audio_feature_dim=7,
        unfreeze_backbone=args.unfreeze_backbone,
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        random_seed=args.seed,
        device=args.device,
        positive_class_weight=positive_class_weight,
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

    val_evaluation, val_predictions = evaluate_checkpoint_on_manifest(
        checkpoint_path=Path(history["best_checkpoint_path"]),
        feature_manifest_path=args.split_manifest,
        split_name=data_config.val_split_name,
        batch_size=args.batch_size,
        device=args.device,
    )
    chosen_eval_threshold = choose_threshold_from_val_predictions(val_predictions)
    train_evaluation, train_predictions = evaluate_checkpoint_on_manifest(
        checkpoint_path=Path(history["best_checkpoint_path"]),
        feature_manifest_path=args.split_manifest,
        split_name=data_config.train_split_name,
        batch_size=args.batch_size,
        threshold=chosen_eval_threshold,
        device=args.device,
    )
    val_evaluation, val_predictions = evaluate_checkpoint_on_manifest(
        checkpoint_path=Path(history["best_checkpoint_path"]),
        feature_manifest_path=args.split_manifest,
        split_name=data_config.val_split_name,
        batch_size=args.batch_size,
        threshold=chosen_eval_threshold,
        device=args.device,
    )
    test_evaluation, test_predictions = evaluate_checkpoint_on_manifest(
        checkpoint_path=Path(history["best_checkpoint_path"]),
        feature_manifest_path=args.split_manifest,
        split_name=data_config.test_split_name,
        batch_size=args.batch_size,
        threshold=chosen_eval_threshold,
        device=args.device,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_predictions.to_csv(output_dir / "train_predictions.csv", index=False)
    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    metrics_table = pd.DataFrame(
        [
            {"split": "train", "epoch": history["best_epoch"], **train_evaluation.metrics.to_dict()},
            {"split": "val", "epoch": history["best_epoch"], **val_evaluation.metrics.to_dict()},
            {"split": "test", "epoch": history["best_epoch"], **test_evaluation.metrics.to_dict()},
        ]
    )
    metrics_table.to_csv(output_dir / "metrics_table.csv", index=False)
    evaluation_summary = {
        "train": train_evaluation.to_dict(),
        "val": val_evaluation.to_dict(),
        "test": test_evaluation.to_dict(),
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(evaluation_summary, indent=2), encoding="utf-8")
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "val_f1": val_evaluation.metrics.f1,
                "val_recall": val_evaluation.metrics.recall,
                "val_precision": val_evaluation.metrics.precision,
                "val_accuracy": val_evaluation.metrics.accuracy,
                "test_f1": test_evaluation.metrics.f1,
                "test_recall": test_evaluation.metrics.recall,
                "test_precision": test_evaluation.metrics.precision,
                "test_accuracy": test_evaluation.metrics.accuracy,
                "chosen_threshold": chosen_eval_threshold,
                "best_epoch": history["best_epoch"],
                "status": "success",
                "error": "",
                "run_name": output_dir.name,
                "output_dir": str(output_dir),
                "best_checkpoint_path": history["best_checkpoint_path"],
                "confusion_matrix": {
                    "train": train_evaluation.to_dict()["confusion_matrix"],
                    "val": val_evaluation.to_dict()["confusion_matrix"],
                    "test": test_evaluation.to_dict()["confusion_matrix"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "run_name": "cnn_lstm_audio_highlight_model",
                "best_epoch": history["best_epoch"],
                "best_checkpoint_path": history["best_checkpoint_path"],
                "latest_checkpoint_path": history["latest_checkpoint_path"],
                "history_path": history["history_path"],
                "monitor_metric": args.monitor_metric,
                "best_metric_value": history["best_metric_value"],
                "feature_columns": list(bundle.feature_names),
                "feature_count": len(bundle.feature_names),
                "positive_class_weight": positive_class_weight,
                "decision_threshold": chosen_eval_threshold,
                "training_config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "lstm_hidden_dim": 256,
                    "dropout": 0.5,
                    "patience": args.patience,
                    "seed": args.seed,
                    "device": args.device,
                    "unfreeze_backbone": args.unfreeze_backbone,
                },
                "best_epoch_metrics": {
                    "train": train_evaluation.metrics.to_dict(),
                    "val": val_evaluation.metrics.to_dict(),
                    "test": test_evaluation.metrics.to_dict(),
                },
                "confusion_matrix": {
                    "train": train_evaluation.to_dict()["confusion_matrix"],
                    "val": val_evaluation.to_dict()["confusion_matrix"],
                    "test": test_evaluation.to_dict()["confusion_matrix"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Training complete")
    print(
        f"manifest={args.split_manifest.resolve()} "
        f"audio_features={', '.join(bundle.feature_names)} "
        f"device={training_config.device}"
    )
    print(f"positive_class_weight={positive_class_weight}")
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
    print("train metrics=" + json.dumps(train_evaluation.metrics.to_dict(), indent=2))
    print("val metrics=" + json.dumps(val_evaluation.metrics.to_dict(), indent=2))
    print("test metrics=" + json.dumps(test_evaluation.metrics.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
