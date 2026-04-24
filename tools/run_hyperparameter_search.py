#!/usr/bin/env python3
"""Run a grid search over CNN+LSTM+audio training hyperparameters."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from run_real_baseline_training import (  # noqa: E402
    choose_threshold_from_val_predictions,
    sanitize_metric_dict,
)
from vod2video.evaluation import evaluate_checkpoint_on_manifest  # noqa: E402
from vod2video.models import build_model  # noqa: E402
from vod2video.training import train_model  # noqa: E402
from vod2video.training_config import CheckpointConfig, DataConfig, ModelConfig, TrainingConfig  # noqa: E402
from vod2video.training_data import (  # noqa: E402
    TrainingDataError,
    build_video_audio_dataloaders_from_manifest,
    compute_positive_class_weight_from_manifest,
)


EPOCH_GRID = [50, 75, 100]
LEARNING_RATE_GRID = [0.00005, 0.0001, 0.0002]
BATCH_SIZE_GRID = [4, 8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an 18-combination hyperparameter grid search for the CNN+LSTM+audio model."
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=REPO_ROOT / "artifacts" / "features" / "branch_2a" / "clip_features.csv",
        help="Path to the real feature manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "hyperparameter_search_v2",
        help="Directory where grid-search outputs will be written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string. Defaults to CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    return parser.parse_args()


def format_learning_rate(value: float) -> str:
    return f"{value:g}"


def build_run_name(*, epochs: int, learning_rate: float, batch_size: int) -> str:
    return f"e{epochs}_lr{format_learning_rate(learning_rate)}_bs{batch_size}"


def metrics_with_prefix(prefix: str, metrics: dict[str, float | int | None]) -> dict[str, float | int | None]:
    return {
        f"{prefix}_f1": metrics.get("f1"),
        f"{prefix}_recall": metrics.get("recall"),
        f"{prefix}_precision": metrics.get("precision"),
        f"{prefix}_accuracy": metrics.get("accuracy"),
    }


def format_metric(value: object, *, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def run_combination(
    *,
    feature_manifest_path: Path,
    run_output_dir: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
    num_workers: int,
    positive_class_weight: float | None,
) -> dict[str, Any]:
    data_config = DataConfig(
        split_manifest_path=feature_manifest_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )
    bundle = build_video_audio_dataloaders_from_manifest(data_config)

    model_config = ModelConfig(
        model_name="cnn_lstm_audio",
        dropout=0.3,
        lstm_hidden_dim=256,
        audio_feature_dim=7,
        unfreeze_backbone=False,
    )
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        weight_decay=0.0,
        epochs=epochs,
        random_seed=42,
        device=device,
        positive_class_weight=positive_class_weight,
        checkpoint=CheckpointConfig(
            output_dir=run_output_dir,
            monitor_metric="f1",
            monitor_mode="max",
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
    chosen_threshold = choose_threshold_from_val_predictions(val_predictions)
    val_evaluation, val_predictions = evaluate_checkpoint_on_manifest(
        checkpoint_path=best_checkpoint_path,
        feature_manifest_path=feature_manifest_path,
        split_name=data_config.val_split_name,
        batch_size=batch_size,
        threshold=chosen_threshold,
        device=device,
    )
    test_evaluation, test_predictions = evaluate_checkpoint_on_manifest(
        checkpoint_path=best_checkpoint_path,
        feature_manifest_path=feature_manifest_path,
        split_name=data_config.test_split_name,
        batch_size=batch_size,
        threshold=chosen_threshold,
        device=device,
    )

    run_output_dir.mkdir(parents=True, exist_ok=True)
    val_predictions.to_csv(run_output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(run_output_dir / "test_predictions.csv", index=False)

    val_metrics = sanitize_metric_dict(val_evaluation.metrics.to_dict())
    test_metrics = sanitize_metric_dict(test_evaluation.metrics.to_dict())
    result = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        **metrics_with_prefix("val", val_metrics),
        **metrics_with_prefix("test", test_metrics),
        "chosen_threshold": float(chosen_threshold),
        "best_epoch": int(history["best_epoch"]),
        "status": "success",
        "error": "",
        "run_name": build_run_name(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size),
        "output_dir": str(run_output_dir),
        "best_checkpoint_path": str(best_checkpoint_path),
    }

    (run_output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_completed_result(
    *,
    result_path: Path,
    run_name: str,
    run_output_dir: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"{result_path} did not contain a JSON object")

    # Keep resumed rows usable even if they were produced by an older version
    # of the script that omitted one of the identifying fields.
    result.setdefault("epochs", epochs)
    result.setdefault("learning_rate", learning_rate)
    result.setdefault("batch_size", batch_size)
    result.setdefault("status", "success")
    result.setdefault("error", "")
    result.setdefault("run_name", run_name)
    result.setdefault("output_dir", str(run_output_dir))
    return result


def save_results_table(results: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    columns = [
        "epochs",
        "learning_rate",
        "batch_size",
        "val_f1",
        "val_recall",
        "val_precision",
        "val_accuracy",
        "test_f1",
        "test_recall",
        "test_precision",
        "test_accuracy",
        "chosen_threshold",
        "best_epoch",
        "status",
        "error",
        "run_name",
        "output_dir",
        "best_checkpoint_path",
    ]
    frame = pd.DataFrame(results)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        frame = frame.reindex(columns=columns)
        frame = frame.sort_values(["val_f1", "val_recall"], ascending=[False, False], na_position="last")
    frame.to_csv(output_dir / "results.csv", index=False)
    return frame


def plot_metric_heatmaps(results: pd.DataFrame, plots_dir: Path, *, metric: str, label: str) -> None:
    for epochs in EPOCH_GRID:
        subset = results.loc[results["epochs"] == epochs]
        pivot = subset.pivot_table(
            index="learning_rate",
            columns="batch_size",
            values=metric,
            aggfunc="mean",
        ).reindex(index=LEARNING_RATE_GRID, columns=BATCH_SIZE_GRID)

        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
        ax.set_title(f"{label} by Learning Rate and Batch Size ({epochs} Epochs)")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Learning Rate")
        ax.set_xticks(range(len(BATCH_SIZE_GRID)), [str(value) for value in BATCH_SIZE_GRID])
        ax.set_yticks(range(len(LEARNING_RATE_GRID)), [format_learning_rate(value) for value in LEARNING_RATE_GRID])
        for row_index, learning_rate in enumerate(LEARNING_RATE_GRID):
            for column_index, batch_size in enumerate(BATCH_SIZE_GRID):
                value = pivot.loc[learning_rate, batch_size]
                annotation = "n/a" if pd.isna(value) else f"{value:.3f}"
                ax.text(column_index, row_index, annotation, ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric}_heatmap_e{epochs}.png", dpi=160)
        plt.close(fig)


def plot_top_10_val_f1(results: pd.DataFrame, plots_dir: Path) -> None:
    top_10 = results.sort_values(["val_f1", "val_recall"], ascending=[False, False]).head(10)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(top_10["run_name"], top_10["val_f1"], color="#2f6f8f")
    ax.set_title("Top 10 Hyperparameter Combinations by Validation F1")
    ax.set_xlabel("Combination")
    ax.set_ylabel("Validation F1")
    ax.set_ylim(0, max(1.0, float(top_10["val_f1"].max()) * 1.1))
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plots_dir / "top_10_val_f1.png", dpi=160)
    plt.close(fig)


def plot_precision_recall_scatter(results: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for learning_rate in LEARNING_RATE_GRID:
        subset = results.loc[results["learning_rate"] == learning_rate]
        if subset.empty:
            continue
        ax.scatter(
            subset["val_recall"],
            subset["val_precision"],
            s=80,
            alpha=0.8,
            label=f"lr={format_learning_rate(learning_rate)}",
        )
    ax.set_title("Validation Recall vs Precision")
    ax.set_xlabel("Validation Recall")
    ax.set_ylabel("Validation Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="Learning Rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "val_recall_vs_precision.png", dpi=160)
    plt.close(fig)


def plot_val_f1_by_epochs(results: pd.DataFrame, plots_dir: Path) -> None:
    averaged = (
        results.groupby(["learning_rate", "epochs"], as_index=False)["val_f1"]
        .mean()
        .sort_values(["learning_rate", "epochs"])
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for learning_rate in LEARNING_RATE_GRID:
        subset = averaged.loc[averaged["learning_rate"] == learning_rate]
        if subset.empty:
            continue
        ax.plot(
            subset["epochs"],
            subset["val_f1"],
            marker="o",
            linewidth=2,
            label=f"lr={format_learning_rate(learning_rate)}",
        )
    ax.set_title("Validation F1 vs Epochs, Averaged Across Batch Sizes")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Validation F1")
    ax.set_xticks(EPOCH_GRID)
    ax.set_ylim(0, max(1.0, float(averaged["val_f1"].max()) * 1.1))
    ax.legend(title="Learning Rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "val_f1_vs_epochs_avg_by_batch_size.png", dpi=160)
    plt.close(fig)


def generate_plots(results: pd.DataFrame, output_dir: Path) -> None:
    successful_results = results.loc[results.get("status", "success") == "success"].copy()
    if successful_results.empty:
        print("No successful runs; skipping plots.", file=sys.stderr)
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    plot_metric_heatmaps(successful_results, plots_dir, metric="val_f1", label="Validation F1")
    plot_metric_heatmaps(successful_results, plots_dir, metric="val_recall", label="Validation Recall")
    plot_top_10_val_f1(successful_results, plots_dir)
    plot_precision_recall_scatter(successful_results, plots_dir)
    plot_val_f1_by_epochs(successful_results, plots_dir)


def print_top_5(results: pd.DataFrame) -> None:
    successful_results = results.loc[results.get("status", "success") == "success"].copy()
    if successful_results.empty:
        print("No successful combinations to rank.")
        return
    top_5 = successful_results.sort_values(["val_f1", "val_recall"], ascending=[False, False]).head(5)
    display_columns = [
        "epochs",
        "learning_rate",
        "batch_size",
        "val_f1",
        "val_recall",
        "val_precision",
        "val_accuracy",
        "test_f1",
        "best_epoch",
        "chosen_threshold",
    ]
    print("Top 5 combinations by val F1 (val recall tiebreaker):")
    print(top_5.loc[:, display_columns].to_string(index=False))


def main() -> int:
    args = parse_args()
    feature_manifest_path = args.feature_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        positive_class_weight = compute_positive_class_weight_from_manifest(feature_manifest_path, cap=10.0)
    except (FileNotFoundError, TrainingDataError, ValueError) as exc:
        print(f"Could not resolve positive class weight: {exc}", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    combinations = list(itertools.product(EPOCH_GRID, LEARNING_RATE_GRID, BATCH_SIZE_GRID))
    total = len(combinations)

    print(f"Starting hyperparameter search with {total} combinations")
    print(f"feature manifest={feature_manifest_path}")
    print(f"output dir={output_dir}")
    print(f"device={args.device}")

    for index, (epochs, learning_rate, batch_size) in enumerate(combinations, start=1):
        run_name = build_run_name(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
        run_output_dir = output_dir / run_name
        result_path = run_output_dir / "result.json"

        if result_path.exists():
            try:
                result = load_completed_result(
                    result_path=result_path,
                    run_name=run_name,
                    run_output_dir=run_output_dir,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                failure = {
                    "run_name": run_name,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "error": f"Could not load existing result.json: {exc}",
                    "status": "failed",
                    "output_dir": str(run_output_dir),
                }
                failures.append(failure)
                results.append(failure)
                print(f"[{index}/{total}] FAILED {run_name}: {failure['error']}", file=sys.stderr)
                continue

            results.append(result)
            print(f"[{index}/{total}] skipping {run_name} (already completed)")
            continue

        try:
            result = run_combination(
                feature_manifest_path=feature_manifest_path,
                run_output_dir=run_output_dir,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                device=args.device,
                num_workers=args.num_workers,
                positive_class_weight=positive_class_weight,
            )
        except Exception as exc:  # noqa: BLE001 - a failed run should not stop the grid.
            failure = {
                "run_name": run_name,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "error": str(exc),
                "status": "failed",
                "output_dir": str(run_output_dir),
            }
            failures.append(failure)
            results.append(failure)
            print(f"[{index}/{total}] FAILED {run_name}: {exc}", file=sys.stderr)
            (run_output_dir / "error.log").parent.mkdir(parents=True, exist_ok=True)
            (run_output_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            continue

        results.append(result)
        print(
            f"[{index}/{total}] completed {run_name}: "
            f"val_f1={format_metric(result['val_f1'])} "
            f"val_recall={format_metric(result['val_recall'])} "
            f"val_precision={format_metric(result['val_precision'])} "
            f"test_f1={format_metric(result['test_f1'])} "
            f"threshold={format_metric(result['chosen_threshold'], digits=3)} "
            f"best_epoch={result['best_epoch']}"
        )

    results_frame = save_results_table(results, output_dir)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
        print(f"Failures recorded in {output_dir / 'failures.csv'}")

    print_top_5(results_frame)
    generate_plots(results_frame, output_dir)
    print(f"Results table saved to {output_dir / 'results.csv'}")
    print(f"Plots saved to {output_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
