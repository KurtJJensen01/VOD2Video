"""Feature subset experiment helpers for Branch 4A."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from .clip_features import DEFAULT_FEATURE_COLUMNS
from .evaluation import evaluate_checkpoint_on_manifest
from .models import build_model
from .training import train_model
from .training_config import CheckpointConfig, DataConfig, ModelConfig, TrainingConfig
from .training_data import TrainingDataError, build_dataloaders_from_manifest, load_split_manifest

VISUAL_FEATURE_COLUMNS = DEFAULT_FEATURE_COLUMNS[:17]
AUDIO_FEATURE_COLUMNS = DEFAULT_FEATURE_COLUMNS[17:]
METADATA_FEATURE_COLUMNS = (
    "start_time_seconds",
    "end_time_seconds",
    "duration_seconds",
    "segment_index",
    "vod_index",
)


@dataclass(frozen=True)
class FeatureGroup:
    """Named feature group used to build repeatable experiment subsets."""

    name: str
    description: str
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class FeatureExperimentSpec:
    """One feature subset experiment definition."""

    name: str
    description: str
    group_names: tuple[str, ...]


def sanitize_metric_dict(metrics: dict[str, float | int]) -> dict[str, float | int | None]:
    cleaned: dict[str, float | int | None] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def build_feature_groups() -> dict[str, FeatureGroup]:
    return {
        "visual_features": FeatureGroup(
            name="visual_features",
            description="Clip-derived visual activity and frame statistics from Phase 2A.",
            feature_names=tuple(VISUAL_FEATURE_COLUMNS),
        ),
        "audio_features": FeatureGroup(
            name="audio_features",
            description="ffmpeg-derived audio summary features from Phase 2A.",
            feature_names=tuple(AUDIO_FEATURE_COLUMNS),
        ),
        "metadata_features": FeatureGroup(
            name="metadata_features",
            description="Lightweight timing and clip identity context derived from the manifest.",
            feature_names=tuple(METADATA_FEATURE_COLUMNS),
        ),
    }


def build_default_experiments() -> tuple[FeatureExperimentSpec, ...]:
    return (
        FeatureExperimentSpec(
            name="visual_only",
            description="Use only visual clip features.",
            group_names=("visual_features",),
        ),
        FeatureExperimentSpec(
            name="audio_only",
            description="Use only audio clip features.",
            group_names=("audio_features",),
        ),
        FeatureExperimentSpec(
            name="visual_audio",
            description="Use the same visual+audio clip feature set as the Branch 3A baseline.",
            group_names=("visual_features", "audio_features"),
        ),
        FeatureExperimentSpec(
            name="metadata_only",
            description="Use only simple manifest-derived timing/context features.",
            group_names=("metadata_features",),
        ),
        FeatureExperimentSpec(
            name="visual_metadata",
            description="Combine visual clip features with simple timing/context features.",
            group_names=("visual_features", "metadata_features"),
        ),
        FeatureExperimentSpec(
            name="all_features",
            description="Combine visual, audio, and metadata-like features.",
            group_names=("visual_features", "audio_features", "metadata_features"),
        ),
    )


def resolve_experiment_feature_names(
    experiment: FeatureExperimentSpec,
    feature_groups: dict[str, FeatureGroup],
) -> tuple[str, ...]:
    feature_names: list[str] = []
    for group_name in experiment.group_names:
        if group_name not in feature_groups:
            raise KeyError(f"Unknown feature group: {group_name}")
        for feature_name in feature_groups[group_name].feature_names:
            if feature_name not in feature_names:
                feature_names.append(feature_name)
    return tuple(feature_names)


def resolve_positive_class_weight(
    manifest_path: Path,
    *,
    train_split_name: str = "train",
) -> float:
    manifest = load_split_manifest(manifest_path)
    if "split" not in manifest.columns or "label" not in manifest.columns:
        raise TrainingDataError("Feature manifest must contain split and label columns.")

    train_rows = manifest.loc[manifest["split"] == train_split_name].copy()
    if train_rows.empty:
        raise TrainingDataError(f"Feature manifest split '{train_split_name}' is empty.")

    label_counts = train_rows["label"].value_counts().to_dict()
    positive_count = int(label_counts.get(1, 0))
    negative_count = int(label_counts.get(0, 0))
    if positive_count == 0:
        raise TrainingDataError("Train split contains no positive examples; cannot derive class weight.")
    return negative_count / positive_count


def load_baseline_summary(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None

    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        return None

    return json.loads(resolved_path.read_text(encoding="utf-8"))


def run_feature_experiment(
    *,
    experiment: FeatureExperimentSpec,
    feature_groups: dict[str, FeatureGroup],
    feature_manifest_path: Path,
    run_output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    seed: int,
    device: str,
    monitor_metric: str,
    positive_class_weight: float | None,
) -> dict[str, object]:
    feature_names = resolve_experiment_feature_names(experiment, feature_groups)
    data_config = DataConfig(
        split_manifest_path=feature_manifest_path,
        batch_size=batch_size,
    )
    bundle = build_dataloaders_from_manifest(
        data_config,
        feature_names=feature_names,
    )

    model_config = ModelConfig(
        input_dim=bundle.input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        random_seed=seed,
        device=device,
        positive_class_weight=positive_class_weight,
        checkpoint=CheckpointConfig(
            output_dir=run_output_dir,
            monitor_metric=monitor_metric,
            monitor_mode="min" if monitor_metric == "loss" else "max",
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

    best_checkpoint_path = Path(str(history["best_checkpoint_path"]))
    val_evaluation, val_predictions = evaluate_checkpoint_on_manifest(
        checkpoint_path=best_checkpoint_path,
        feature_manifest_path=feature_manifest_path,
        split_name=data_config.val_split_name,
        batch_size=batch_size,
        device=device,
    )
    test_evaluation, test_predictions = evaluate_checkpoint_on_manifest(
        checkpoint_path=best_checkpoint_path,
        feature_manifest_path=feature_manifest_path,
        split_name=data_config.test_split_name,
        batch_size=batch_size,
        device=device,
    )

    val_payload = val_evaluation.to_dict()
    test_payload = test_evaluation.to_dict()
    best_epoch = int(history["best_epoch"])
    epoch_record = next(record for record in history["epochs"] if int(record["epoch"]) == best_epoch)

    run_output_dir.mkdir(parents=True, exist_ok=True)
    val_predictions.to_csv(run_output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(run_output_dir / "test_predictions.csv", index=False)

    summary = {
        "experiment_name": experiment.name,
        "description": experiment.description,
        "feature_groups": list(experiment.group_names),
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "output_dir": str(run_output_dir.expanduser().resolve()),
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path.expanduser().resolve()),
        "latest_checkpoint_path": history["latest_checkpoint_path"],
        "history_path": history["history_path"],
        "monitor_metric": monitor_metric,
        "best_metric_value": history["best_metric_value"],
        "positive_class_weight": positive_class_weight,
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "seed": seed,
            "device": device,
        },
        "best_epoch_metrics": {
            "train": sanitize_metric_dict(epoch_record["train"]),
            "val": sanitize_metric_dict(val_payload["metrics"]),
            "test": sanitize_metric_dict(test_payload["metrics"]),
        },
        "confusion_matrix": {
            "val": val_payload["confusion_matrix"],
            "test": test_payload["confusion_matrix"],
        },
        "normalization_stats": bundle.normalization_stats,
    }
    (run_output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def build_feature_experiment_table(
    results: Iterable[dict[str, object]],
    *,
    baseline_summary: dict[str, object] | None = None,
    rank_metric: str = "val_f1",
) -> pd.DataFrame:
    baseline_val_metrics = None
    baseline_test_metrics = None
    baseline_name = None
    if baseline_summary is not None:
        baseline_name = str(baseline_summary.get("run_name", "baseline"))
        best_epoch_metrics = baseline_summary.get("best_epoch_metrics", {})
        if isinstance(best_epoch_metrics, dict):
            baseline_val_metrics = best_epoch_metrics.get("val")
            baseline_test_metrics = best_epoch_metrics.get("test")

    rows: list[dict[str, object]] = []
    for result in results:
        best_epoch_metrics = result["best_epoch_metrics"]
        val_metrics = best_epoch_metrics["val"]
        test_metrics = best_epoch_metrics["test"]

        row = {
            "experiment_name": result["experiment_name"],
            "description": result["description"],
            "feature_groups": "|".join(result["feature_groups"]),
            "feature_count": result["feature_count"],
            "best_epoch": result["best_epoch"],
            "monitor_metric": result["monitor_metric"],
            "best_metric_value": result["best_metric_value"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "best_checkpoint_path": result["best_checkpoint_path"],
        }
        if isinstance(baseline_val_metrics, dict):
            row["delta_vs_baseline_val_f1"] = _safe_metric_delta(
                row["val_f1"],
                baseline_val_metrics.get("f1"),
            )
        if isinstance(baseline_test_metrics, dict):
            row["delta_vs_baseline_test_f1"] = _safe_metric_delta(
                row["test_f1"],
                baseline_test_metrics.get("f1"),
            )
        if baseline_name is not None:
            row["baseline_name"] = baseline_name
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    sort_column = rank_metric if rank_metric in table.columns else "val_f1"
    table = table.sort_values(
        by=[sort_column, "test_f1", "feature_count", "experiment_name"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def build_feature_experiment_summary(
    *,
    feature_manifest_path: Path,
    feature_groups: dict[str, FeatureGroup],
    results: list[dict[str, object]],
    comparison_table: pd.DataFrame,
    baseline_summary: dict[str, object] | None,
    output_dir: Path,
) -> dict[str, object]:
    best_experiment = comparison_table.iloc[0].to_dict() if not comparison_table.empty else None
    return {
        "branch": "4a_feature_improvement",
        "feature_manifest_path": str(feature_manifest_path.expanduser().resolve()),
        "output_dir": str(output_dir.expanduser().resolve()),
        "feature_groups": {
            name: asdict(group)
            for name, group in feature_groups.items()
        },
        "experiment_count": len(results),
        "experiments": results,
        "comparison_table_path": str((output_dir / "feature_experiment_table.csv").expanduser().resolve()),
        "best_experiment": best_experiment,
        "baseline_comparison": baseline_summary,
    }


def write_feature_experiment_outputs(
    *,
    output_dir: Path,
    comparison_table: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, Path]:
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "table_csv": resolved_output_dir / "feature_experiment_table.csv",
        "summary_json": resolved_output_dir / "feature_experiment_summary.json",
    }
    comparison_table.to_csv(paths["table_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return paths


def _safe_metric_delta(current_value: object, baseline_value: object) -> float | None:
    try:
        current_float = float(current_value)
        baseline_float = float(baseline_value)
    except (TypeError, ValueError):
        return None
    if math.isnan(current_float) or math.isnan(baseline_float):
        return None
    return current_float - baseline_float
