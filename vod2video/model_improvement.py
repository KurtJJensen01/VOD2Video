"""Model improvement experiment helpers for Branch 4B."""

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
from .feature_improvement import load_baseline_summary, resolve_positive_class_weight
from .models import build_model
from .training import train_model
from .training_config import CheckpointConfig, DataConfig, ModelConfig, TrainingConfig
from .training_data import TrainingDataError, build_dataloaders_from_manifest


@dataclass(frozen=True)
class ModelExperimentSpec:
    """One practical Branch 4B experiment definition."""

    name: str
    description: str
    epochs: int | None = None
    learning_rate: float | None = None
    hidden_dim: int | None = None
    dropout: float | None = None
    positive_class_weight: float | None = None
    disable_class_weight: bool = False


def sanitize_metric_dict(metrics: dict[str, float | int]) -> dict[str, float | int | None]:
    cleaned: dict[str, float | int | None] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def build_default_model_experiments() -> tuple[ModelExperimentSpec, ...]:
    return (
        ModelExperimentSpec(
            name="baseline_config",
            description="Reuse the current baseline training setup with the best feature subset.",
        ),
        ModelExperimentSpec(
            name="more_epochs",
            description="Check whether a longer training schedule improves the selected setup.",
            epochs=20,
        ),
        ModelExperimentSpec(
            name="lower_learning_rate",
            description="Use a smaller learning rate for steadier optimization.",
            learning_rate=5e-4,
        ),
        ModelExperimentSpec(
            name="larger_hidden_dim",
            description="Increase hidden width while keeping the rest of the setup simple.",
            hidden_dim=64,
        ),
        ModelExperimentSpec(
            name="lower_dropout",
            description="Reduce dropout to see whether the model benefits from less regularization.",
            dropout=0.05,
        ),
        ModelExperimentSpec(
            name="no_class_weight",
            description="Disable positive-class weighting to measure its actual impact.",
            disable_class_weight=True,
        ),
        ModelExperimentSpec(
            name="best_guess_combo",
            description="A single combined tuning guess using longer training, lower LR, larger hidden layer, and lower dropout.",
            epochs=20,
            learning_rate=5e-4,
            hidden_dim=64,
            dropout=0.05,
        ),
    )


def resolve_feature_columns_for_model_improvement(
    *,
    feature_improvement_summary_path: Path | None,
    baseline_summary_path: Path | None,
) -> tuple[list[str], dict[str, object] | None]:
    feature_summary = load_baseline_summary(feature_improvement_summary_path)
    if isinstance(feature_summary, dict):
        best_experiment = feature_summary.get("best_experiment")
        experiment_payloads = feature_summary.get("experiments")
        if isinstance(best_experiment, dict) and isinstance(experiment_payloads, list):
            best_name = best_experiment.get("experiment_name")
            for experiment in experiment_payloads:
                if (
                    isinstance(experiment, dict)
                    and experiment.get("experiment_name") == best_name
                    and isinstance(experiment.get("feature_names"), list)
                ):
                    return [str(name) for name in experiment["feature_names"]], feature_summary

    baseline_summary = load_baseline_summary(baseline_summary_path)
    if isinstance(baseline_summary, dict) and isinstance(baseline_summary.get("feature_columns"), list):
        return [str(name) for name in baseline_summary["feature_columns"]], baseline_summary

    return list(DEFAULT_FEATURE_COLUMNS), None


def run_model_experiment(
    *,
    experiment: ModelExperimentSpec,
    feature_manifest_path: Path,
    feature_names: Iterable[str],
    run_output_dir: Path,
    batch_size: int,
    weight_decay: float,
    seed: int,
    device: str,
    monitor_metric: str,
    base_epochs: int,
    base_learning_rate: float,
    base_hidden_dim: int,
    base_dropout: float,
    base_positive_class_weight: float | None,
) -> dict[str, object]:
    feature_name_list = list(feature_names)
    data_config = DataConfig(
        split_manifest_path=feature_manifest_path,
        batch_size=batch_size,
    )
    bundle = build_dataloaders_from_manifest(
        data_config,
        feature_names=feature_name_list,
    )

    resolved_epochs = int(experiment.epochs if experiment.epochs is not None else base_epochs)
    resolved_learning_rate = float(
        experiment.learning_rate if experiment.learning_rate is not None else base_learning_rate
    )
    resolved_hidden_dim = int(experiment.hidden_dim if experiment.hidden_dim is not None else base_hidden_dim)
    resolved_dropout = float(experiment.dropout if experiment.dropout is not None else base_dropout)
    if experiment.disable_class_weight:
        resolved_positive_class_weight = None
    elif experiment.positive_class_weight is not None:
        resolved_positive_class_weight = float(experiment.positive_class_weight)
    else:
        resolved_positive_class_weight = base_positive_class_weight

    model_config = ModelConfig(
        input_dim=bundle.input_dim,
        hidden_dim=resolved_hidden_dim,
        dropout=resolved_dropout,
    )
    training_config = TrainingConfig(
        learning_rate=resolved_learning_rate,
        weight_decay=weight_decay,
        epochs=resolved_epochs,
        random_seed=seed,
        device=device,
        positive_class_weight=resolved_positive_class_weight,
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

    best_epoch = int(history["best_epoch"])
    epoch_record = next(record for record in history["epochs"] if int(record["epoch"]) == best_epoch)
    val_payload = val_evaluation.to_dict()
    test_payload = test_evaluation.to_dict()

    run_output_dir.mkdir(parents=True, exist_ok=True)
    val_predictions.to_csv(run_output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(run_output_dir / "test_predictions.csv", index=False)

    summary = {
        "experiment_name": experiment.name,
        "description": experiment.description,
        "feature_names": feature_name_list,
        "feature_count": len(feature_name_list),
        "output_dir": str(run_output_dir.expanduser().resolve()),
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path.expanduser().resolve()),
        "latest_checkpoint_path": history["latest_checkpoint_path"],
        "history_path": history["history_path"],
        "monitor_metric": monitor_metric,
        "best_metric_value": history["best_metric_value"],
        "positive_class_weight": resolved_positive_class_weight,
        "class_weight_enabled": resolved_positive_class_weight is not None,
        "training_config": {
            "epochs": resolved_epochs,
            "batch_size": batch_size,
            "learning_rate": resolved_learning_rate,
            "weight_decay": weight_decay,
            "hidden_dim": resolved_hidden_dim,
            "dropout": resolved_dropout,
            "seed": seed,
            "device": device,
        },
        "overrides": {
            "epochs": experiment.epochs,
            "learning_rate": experiment.learning_rate,
            "hidden_dim": experiment.hidden_dim,
            "dropout": experiment.dropout,
            "positive_class_weight": experiment.positive_class_weight,
            "disable_class_weight": experiment.disable_class_weight,
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


def build_model_experiment_table(
    results: Iterable[dict[str, object]],
    *,
    baseline_summary: dict[str, object] | None = None,
    rank_metric: str = "val_f1",
) -> pd.DataFrame:
    baseline_val_metrics = None
    baseline_test_metrics = None
    baseline_name = None
    if baseline_summary is not None:
        baseline_name = str(baseline_summary.get("run_name", baseline_summary.get("branch", "baseline")))
        best_epoch_metrics = baseline_summary.get("best_epoch_metrics", {})
        if isinstance(best_epoch_metrics, dict):
            baseline_val_metrics = best_epoch_metrics.get("val")
            baseline_test_metrics = best_epoch_metrics.get("test")

    rows: list[dict[str, object]] = []
    for result in results:
        best_epoch_metrics = result["best_epoch_metrics"]
        training_config = result["training_config"]
        val_metrics = best_epoch_metrics["val"]
        test_metrics = best_epoch_metrics["test"]
        row = {
            "experiment_name": result["experiment_name"],
            "description": result["description"],
            "feature_count": result["feature_count"],
            "best_epoch": result["best_epoch"],
            "monitor_metric": result["monitor_metric"],
            "best_metric_value": result["best_metric_value"],
            "epochs": training_config["epochs"],
            "learning_rate": training_config["learning_rate"],
            "hidden_dim": training_config["hidden_dim"],
            "dropout": training_config["dropout"],
            "class_weight_enabled": result["class_weight_enabled"],
            "positive_class_weight": result["positive_class_weight"],
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
            row["delta_vs_baseline_val_f1"] = _safe_metric_delta(row["val_f1"], baseline_val_metrics.get("f1"))
        if isinstance(baseline_test_metrics, dict):
            row["delta_vs_baseline_test_f1"] = _safe_metric_delta(row["test_f1"], baseline_test_metrics.get("f1"))
        if baseline_name is not None:
            row["baseline_name"] = baseline_name
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    sort_column = rank_metric if rank_metric in table.columns else "val_f1"
    table = table.sort_values(
        by=[sort_column, "test_f1", "val_precision", "experiment_name"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def build_model_experiment_summary(
    *,
    feature_manifest_path: Path,
    feature_columns: list[str],
    feature_setup_source: dict[str, object] | None,
    results: list[dict[str, object]],
    comparison_table: pd.DataFrame,
    baseline_summary: dict[str, object] | None,
    output_dir: Path,
) -> dict[str, object]:
    best_experiment = comparison_table.iloc[0].to_dict() if not comparison_table.empty else None
    return {
        "branch": "4b_model_improvement",
        "feature_manifest_path": str(feature_manifest_path.expanduser().resolve()),
        "output_dir": str(output_dir.expanduser().resolve()),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "feature_setup_source": feature_setup_source,
        "experiment_count": len(results),
        "experiments": results,
        "comparison_table_path": str((output_dir / "model_experiment_table.csv").expanduser().resolve()),
        "best_experiment": best_experiment,
        "baseline_comparison": baseline_summary,
    }


def write_model_experiment_outputs(
    *,
    output_dir: Path,
    comparison_table: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, Path]:
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "table_csv": resolved_output_dir / "model_experiment_table.csv",
        "summary_json": resolved_output_dir / "model_experiment_summary.json",
    }
    comparison_table.to_csv(paths["table_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return paths


def build_feature_setup_source_payload(
    feature_setup_source: dict[str, object] | None,
) -> dict[str, object] | None:
    if feature_setup_source is None:
        return None

    branch = feature_setup_source.get("branch")
    if branch == "4a_feature_improvement":
        best_experiment = feature_setup_source.get("best_experiment")
        return {
            "type": "branch_4a_best_experiment",
            "branch": branch,
            "best_experiment": best_experiment,
        }

    return {
        "type": "baseline_summary_fallback",
        "run_name": feature_setup_source.get("run_name"),
        "feature_count": feature_setup_source.get("feature_count"),
    }


def resolve_base_positive_class_weight(
    *,
    feature_manifest_path: Path,
    explicit_positive_class_weight: float | None,
    disable_auto_class_weight: bool,
) -> float | None:
    if explicit_positive_class_weight is not None:
        return float(explicit_positive_class_weight)
    if disable_auto_class_weight:
        return None
    return resolve_positive_class_weight(feature_manifest_path)


def _safe_metric_delta(current_value: object, baseline_value: object) -> float | None:
    try:
        current_float = float(current_value)
        baseline_float = float(baseline_value)
    except (TypeError, ValueError):
        return None
    if math.isnan(current_float) or math.isnan(baseline_float):
        return None
    return current_float - baseline_float
