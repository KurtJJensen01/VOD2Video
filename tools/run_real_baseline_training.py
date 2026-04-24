#!/usr/bin/env python3
"""Branch 3A real-feature baseline training workflow."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.clip_features import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from vod2video.evaluation import evaluate_checkpoint_on_manifest  # noqa: E402
from vod2video.metrics import sweep_thresholds  # noqa: E402
from vod2video.models import build_model  # noqa: E402
from vod2video.training import train_model  # noqa: E402
from vod2video.training_config import CheckpointConfig, DataConfig, ModelConfig, TrainingConfig  # noqa: E402
from vod2video.training_data import (  # noqa: E402
    TrainingDataError,
    build_video_audio_dataloaders_from_manifest,
    compute_positive_class_weight_from_manifest,
)


DEFAULT_COMPARE_CHECKPOINT = REPO_ROOT / "artifacts" / "training" / "branch_2b_baseline" / "best_model.pt"
DEFAULT_COMPARE_MANIFEST = REPO_ROOT / "artifacts" / "splits" / "branch_1c" / "all_splits.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the first presentation-ready baseline model on the real Phase 2A feature manifest."
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=REPO_ROOT / "artifacts" / "features" / "branch_2a" / "clip_features.csv",
        help="Path to the Phase 2A real feature manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "training" / "branch_3a_real_baseline",
        help="Directory used for checkpoints and Branch 3A summary artifacts.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Optimizer weight decay.")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop training after this many consecutive epochs without validation F1 improvement.",
    )
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer width.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Unfreeze the ResNet18 backbone for experimentation.",
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
        default=list(DEFAULT_FEATURE_COLUMNS),
        help="Feature columns to use from the Phase 2A manifest.",
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
    parser.add_argument(
        "--compare-checkpoint",
        type=Path,
        default=DEFAULT_COMPARE_CHECKPOINT,
        help="Optional earlier baseline checkpoint to compare against.",
    )
    parser.add_argument(
        "--compare-manifest",
        type=Path,
        default=DEFAULT_COMPARE_MANIFEST,
        help="Manifest compatible with the comparison checkpoint.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip the optional comparison artifact.",
    )
    return parser.parse_args()


def resolve_positive_class_weight(args: argparse.Namespace, manifest_path: Path) -> float | None:
    if args.positive_class_weight is not None:
        return float(args.positive_class_weight)
    if args.disable_auto_class_weight:
        return None
    return compute_positive_class_weight_from_manifest(manifest_path, cap=10.0)


def sanitize_metric_dict(metrics: dict[str, float | int]) -> dict[str, float | int | None]:
    cleaned: dict[str, float | int | None] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def build_metrics_table(history: dict[str, object], test_metrics: dict[str, float | int | None]) -> pd.DataFrame:
    best_epoch = int(history["best_epoch"])
    epoch_record = next(record for record in history["epochs"] if int(record["epoch"]) == best_epoch)

    rows = []
    for split_name in ("train", "val"):
        metrics = sanitize_metric_dict(epoch_record[split_name])
        rows.append(
            {
                "split": split_name,
                "epoch": best_epoch,
                **metrics,
            }
        )
    rows.append(
        {
            "split": "test",
            "epoch": best_epoch,
            **test_metrics,
        }
    )
    return pd.DataFrame(rows)


def choose_threshold_from_val_predictions(predictions: pd.DataFrame) -> float:
    probabilities = torch.tensor(
        predictions["predicted_probability"].to_numpy(dtype="float32"),
        dtype=torch.float32,
    ).clamp(1e-6, 1.0 - 1e-6)
    labels = torch.tensor(predictions["label"].to_numpy(dtype="float32"), dtype=torch.float32)
    threshold, _ = sweep_thresholds(torch.logit(probabilities), labels, loss=float("nan"), min_precision=0.15)
    return float(threshold)


def build_run_summary(
    *,
    args: argparse.Namespace,
    history: dict[str, object],
    val_evaluation: dict[str, object],
    test_evaluation: dict[str, object],
    positive_class_weight: float | None,
    feature_manifest_path: Path,
) -> dict[str, object]:
    best_epoch = int(history["best_epoch"])
    epoch_record = next(record for record in history["epochs"] if int(record["epoch"]) == best_epoch)
    return {
        "run_name": "branch_3a_real_baseline",
        "feature_manifest_path": str(feature_manifest_path.expanduser().resolve()),
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "best_epoch": best_epoch,
        "best_checkpoint_path": history["best_checkpoint_path"],
        "latest_checkpoint_path": history["latest_checkpoint_path"],
        "history_path": history["history_path"],
        "monitor_metric": args.monitor_metric,
        "best_metric_value": history["best_metric_value"],
        "feature_columns": list(args.feature_columns),
        "feature_count": 7,
        "positive_class_weight": positive_class_weight,
        "decision_threshold": float(val_evaluation["threshold"]),
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": 0.5,
            "patience": args.patience,
            "seed": args.seed,
            "device": args.device,
            "unfreeze_backbone": args.unfreeze_backbone,
        },
        "best_epoch_metrics": {
            "train": sanitize_metric_dict(epoch_record["train"]),
            "val": val_evaluation["metrics"],
            "test": test_evaluation["metrics"],
        },
        "confusion_matrix": {
            "val": val_evaluation["confusion_matrix"],
            "test": test_evaluation["confusion_matrix"],
        },
    }


def maybe_build_comparison(args: argparse.Namespace, output_dir: Path) -> dict[str, object] | None:
    if args.skip_comparison:
        return None
    compare_checkpoint = args.compare_checkpoint.expanduser().resolve()
    compare_manifest = args.compare_manifest.expanduser().resolve()
    if not compare_checkpoint.exists() or not compare_manifest.exists():
        return None

    evaluation, _ = evaluate_checkpoint_on_manifest(
        checkpoint_path=compare_checkpoint,
        feature_manifest_path=compare_manifest,
        split_name="test",
        batch_size=64,
        device=args.device,
    )
    comparison_summary = {
        "name": "placeholder_feature_baseline",
        "checkpoint_path": str(compare_checkpoint),
        "manifest_path": str(compare_manifest),
        "test_metrics": sanitize_metric_dict(evaluation.metrics.to_dict()),
        "confusion_matrix": evaluation.to_dict()["confusion_matrix"],
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(comparison_summary, indent=2),
        encoding="utf-8",
    )
    return comparison_summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    try:
        positive_class_weight = resolve_positive_class_weight(args, args.feature_manifest)
        data_config = DataConfig(
            split_manifest_path=args.feature_manifest,
            batch_size=args.batch_size,
        )
        bundle = build_video_audio_dataloaders_from_manifest(data_config)
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
            output_dir=output_dir,
            monitor_metric=args.monitor_metric,
            monitor_mode="min" if args.monitor_metric == "loss" else "max",
        ),
    )

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
        feature_manifest_path=args.feature_manifest,
        split_name=data_config.val_split_name,
        batch_size=args.batch_size,
        device=args.device,
    )
    chosen_eval_threshold = choose_threshold_from_val_predictions(val_predictions)
    val_evaluation, val_predictions = evaluate_checkpoint_on_manifest(
        checkpoint_path=Path(history["best_checkpoint_path"]),
        feature_manifest_path=args.feature_manifest,
        split_name=data_config.val_split_name,
        batch_size=args.batch_size,
        threshold=chosen_eval_threshold,
        device=args.device,
    )
    train_evaluation, train_predictions = evaluate_checkpoint_on_manifest(
        checkpoint_path=Path(history["best_checkpoint_path"]),
        feature_manifest_path=args.feature_manifest,
        split_name=data_config.train_split_name,
        batch_size=args.batch_size,
        threshold=chosen_eval_threshold,
        device=args.device,
    )
    test_evaluation, test_predictions = evaluate_checkpoint_on_manifest(
        checkpoint_path=Path(history["best_checkpoint_path"]),
        feature_manifest_path=args.feature_manifest,
        split_name=data_config.test_split_name,
        batch_size=args.batch_size,
        threshold=chosen_eval_threshold,
        device=args.device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_table = build_metrics_table(history, sanitize_metric_dict(test_evaluation.metrics.to_dict()))
    metrics_table.to_csv(output_dir / "metrics_table.csv", index=False)
    train_predictions.to_csv(output_dir / "train_predictions.csv", index=False)
    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    val_payload = val_evaluation.to_dict()
    test_payload = test_evaluation.to_dict()
    run_summary = build_run_summary(
        args=args,
        history=history,
        val_evaluation={
            "metrics": sanitize_metric_dict(val_payload["metrics"]),
            "confusion_matrix": val_payload["confusion_matrix"],
            "threshold": val_payload["threshold"],
        },
        test_evaluation={
            "metrics": sanitize_metric_dict(test_payload["metrics"]),
            "confusion_matrix": test_payload["confusion_matrix"],
        },
        positive_class_weight=positive_class_weight,
        feature_manifest_path=args.feature_manifest,
    )

    (output_dir / "metrics_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "train": {
                    **train_evaluation.to_dict(),
                    "metrics": sanitize_metric_dict(train_evaluation.metrics.to_dict()),
                },
                "val": {
                    **val_payload,
                    "metrics": sanitize_metric_dict(val_payload["metrics"]),
                },
                "test": {
                    **test_payload,
                    "metrics": sanitize_metric_dict(test_payload["metrics"]),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    comparison_summary = maybe_build_comparison(args, output_dir)

    print("Branch 3A baseline run complete")
    print(f"feature manifest={args.feature_manifest.expanduser().resolve()}")
    print(f"output dir={output_dir}")
    print(f"best checkpoint={history['best_checkpoint_path']}")
    print(f"best epoch={history['best_epoch']}")
    print("validation metrics=" + json.dumps(sanitize_metric_dict(val_payload["metrics"])))
    print("test metrics=" + json.dumps(sanitize_metric_dict(test_payload["metrics"])))
    print(f"metrics summary={output_dir / 'metrics_summary.json'}")
    print(f"metrics table={output_dir / 'metrics_table.csv'}")
    print(f"evaluation summary={output_dir / 'evaluation_summary.json'}")
    if comparison_summary is not None:
        print(f"comparison summary={output_dir / 'comparison_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
