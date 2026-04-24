#!/usr/bin/env python3
"""Generate standalone visualizations for a CNN+LSTM+MLP search run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("artifacts") / "visualization" / "cnn_run"
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


class VisualizationError(ValueError):
    """Raised when expected run artifacts are missing or malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CNN+LSTM+MLP run visualizations from metrics/result/prediction artifacts."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Hyperparameter search run folder containing metrics.json, result.json, and prediction CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where outputs will be written. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VisualizationError(f"Could not parse JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VisualizationError(f"Expected {path} to contain a JSON object.")
    return payload


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise VisualizationError(f"Prediction file has no rows: {path}")
    return frame


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def format_metric(value: Any, *, digits: int = 3) -> str:
    numeric = as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{digits}f}"


def require_epochs(metrics_payload: dict[str, Any]) -> list[dict[str, Any]]:
    epochs = metrics_payload.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise VisualizationError("metrics.json must contain a non-empty 'epochs' list.")
    normalized: list[dict[str, Any]] = []
    for item in epochs:
        if not isinstance(item, dict):
            raise VisualizationError("Each metrics.json epoch record must be a JSON object.")
        normalized.append(item)
    return normalized


def metric_from_epoch(epoch: dict[str, Any], split_name: str, metric_name: str) -> float | None:
    split_payload = epoch.get(split_name, {})
    if not isinstance(split_payload, dict):
        return None
    return as_float(split_payload.get(metric_name))


def get_best_epoch_record(metrics_payload: dict[str, Any], epochs: list[dict[str, Any]]) -> dict[str, Any]:
    best_epoch = metrics_payload.get("best_epoch")
    if best_epoch is not None:
        for epoch in epochs:
            if int(epoch.get("epoch", -1)) == int(best_epoch):
                return epoch
    return max(epochs, key=lambda epoch: metric_from_epoch(epoch, "val", "f1") or float("-inf"))


def get_threshold(result_payload: dict[str, Any]) -> float:
    for key in ("chosen_threshold", "decision_threshold", "threshold"):
        value = as_float(result_payload.get(key))
        if value is not None:
            return value
    return 0.5


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...], *, role: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise VisualizationError(
        f"Could not find {role} column. Tried: {', '.join(candidates)}. "
        f"Available columns: {', '.join(frame.columns)}"
    )


def labels_and_probabilities(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    label_col = find_column(frame, ("label", "true_label", "target", "y_true"), role="label")
    probability_col = find_column(
        frame,
        ("predicted_probability", "probability", "probability_score", "score", "positive_probability"),
        role="probability",
    )
    labels = pd.to_numeric(frame[label_col], errors="coerce")
    probabilities = pd.to_numeric(frame[probability_col], errors="coerce")
    valid = labels.notna() & probabilities.notna()
    if not valid.any():
        raise VisualizationError("Prediction file did not contain any valid label/probability pairs.")
    return labels.loc[valid].astype(int), probabilities.loc[valid].astype(float)


def compute_metrics(labels: pd.Series, probabilities: pd.Series, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    positives = labels == 1
    negatives = labels == 0
    predicted_positives = predictions == 1
    predicted_negatives = predictions == 0

    tp = int((positives & predicted_positives).sum())
    tn = int((negatives & predicted_negatives).sum())
    fp = int((negatives & predicted_positives).sum())
    fn = int((positives & predicted_negatives).sum())
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "sample_count": total,
    }


def get_confusion_value(confusion: dict[str, Any], *names: str) -> int:
    for name in names:
        value = confusion.get(name)
        if value is not None:
            return int(value)
    return 0


def normalize_confusion_matrix(confusion: dict[str, Any]) -> dict[str, int]:
    return {
        "true_negative": get_confusion_value(confusion, "true_negative", "true_negatives", "tn"),
        "false_positive": get_confusion_value(confusion, "false_positive", "false_positives", "fp"),
        "false_negative": get_confusion_value(confusion, "false_negative", "false_negatives", "fn"),
        "true_positive": get_confusion_value(confusion, "true_positive", "true_positives", "tp"),
    }


def find_result_confusion(result_payload: dict[str, Any], split_name: str) -> dict[str, int] | None:
    candidates: list[Any] = [
        result_payload.get("confusion_matrix", {}).get(split_name)
        if isinstance(result_payload.get("confusion_matrix"), dict)
        else None,
        result_payload.get(f"{split_name}_confusion_matrix"),
    ]
    split_payload = result_payload.get(split_name)
    if isinstance(split_payload, dict):
        candidates.append(split_payload.get("confusion_matrix"))

    for candidate in candidates:
        if isinstance(candidate, dict):
            return normalize_confusion_matrix(candidate)
    return None


def result_split_metrics(result_payload: dict[str, Any], split_name: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    split_payload = result_payload.get(split_name)
    nested_metrics = split_payload.get("metrics") if isinstance(split_payload, dict) else None
    for metric in METRIC_NAMES:
        value = None
        if isinstance(split_payload, dict):
            value = split_payload.get(metric)
        if value is None and isinstance(nested_metrics, dict):
            value = nested_metrics.get(metric)
        if value is None:
            value = result_payload.get(f"{split_name}_{metric}")
        metrics[metric] = as_float(value)
    return metrics


def metrics_for_table(
    *,
    result_payload: dict[str, Any],
    best_epoch_record: dict[str, Any],
    test_predictions: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for split_name in ("train", "val"):
        split_payload = best_epoch_record.get(split_name, {})
        if not isinstance(split_payload, dict):
            split_payload = {}
        row: dict[str, str] = {"split": split_name}
        for metric in METRIC_NAMES:
            row[metric] = format_metric(split_payload.get(metric))
        rows.append(row)

    test_metrics = result_split_metrics(result_payload, "test")
    if any(value is None for value in test_metrics.values()):
        labels, probabilities = labels_and_probabilities(test_predictions)
        computed_test = compute_metrics(labels, probabilities, threshold)
        test_metrics = {metric: as_float(computed_test.get(metric)) for metric in METRIC_NAMES}
    rows.append({"split": "test", **{metric: format_metric(test_metrics.get(metric)) for metric in METRIC_NAMES}})
    return pd.DataFrame(rows, columns=("split", *METRIC_NAMES))


def save_confusion_matrix_png(confusion: dict[str, int], output_path: Path, *, title: str) -> None:
    values = [
        [confusion["true_negative"], confusion["false_positive"]],
        [confusion["false_negative"], confusion["true_positive"]],
    ]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(values, cmap="Blues")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["non-highlight", "highlight"])
    ax.set_yticks([0, 1], labels=["non-highlight", "highlight"])
    max_value = max(max(row) for row in values) or 1
    for row_index, row in enumerate(values):
        for col_index, value in enumerate(row):
            color = "white" if value > max_value * 0.55 else "#1f2937"
            ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=16, color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_metric_line_png(
    epochs: list[dict[str, Any]],
    output_path: Path,
    *,
    metric_name: str,
    title: str,
    ylabel: str,
) -> None:
    epoch_numbers = [int(epoch.get("epoch", index + 1)) for index, epoch in enumerate(epochs)]
    train_values = [metric_from_epoch(epoch, "train", metric_name) for epoch in epochs]
    val_values = [metric_from_epoch(epoch, "val", metric_name) for epoch in epochs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch_numbers, train_values, marker="o", linewidth=2, label="train", color="#2563eb")
    ax.plot(epoch_numbers, val_values, marker="o", linewidth=2, label="val", color="#f97316")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_table_png(frame: pd.DataFrame, output_path: Path, *, title: str, font_size: int = 10) -> None:
    row_count = max(len(frame), 1)
    fig_height = max(2.4, 1.1 + row_count * 0.42)
    fig_width = max(7.0, len(frame.columns) * 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    table = ax.table(
        cellText=frame.astype(str).values,
        colLabels=list(frame.columns),
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.35)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eef2ff")
        else:
            cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#cbd5e1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_threshold_sweep(val_predictions: pd.DataFrame) -> pd.DataFrame:
    labels, probabilities = labels_and_probabilities(val_predictions)
    rows = []
    for index in range(5, 100, 5):
        metrics = compute_metrics(labels, probabilities, index / 100)
        rows.append(
            {
                "threshold": f"{metrics['threshold']:.2f}",
                "accuracy": format_metric(metrics["accuracy"]),
                "precision": format_metric(metrics["precision"]),
                "recall": format_metric(metrics["recall"]),
                "f1": format_metric(metrics["f1"]),
                "tp": metrics["true_positives"],
                "tn": metrics["true_negatives"],
                "fp": metrics["false_positives"],
                "fn": metrics["false_negatives"],
                "sample_count": metrics["sample_count"],
            }
        )
    return pd.DataFrame(rows)


def save_threshold_sweep_csv(frame: pd.DataFrame, output_path: Path) -> None:
    frame.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def generate_visualizations(run_dir: Path, output_dir: Path) -> dict[str, str]:
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = load_json(run_dir / "metrics.json")
    result_payload = load_json(run_dir / "result.json")
    val_predictions = load_predictions(run_dir / "val_predictions.csv")
    test_predictions = load_predictions(run_dir / "test_predictions.csv")

    epochs = require_epochs(metrics_payload)
    best_epoch_record = get_best_epoch_record(metrics_payload, epochs)
    threshold = get_threshold(result_payload)

    artifacts: dict[str, str] = {}
    for split_name, predictions in (("val", val_predictions), ("test", test_predictions)):
        confusion = find_result_confusion(result_payload, split_name)
        if confusion is None:
            labels, probabilities = labels_and_probabilities(predictions)
            computed = compute_metrics(labels, probabilities, threshold)
            confusion = normalize_confusion_matrix(computed)
        output_path = output_dir / f"{split_name}_confusion_matrix.png"
        save_confusion_matrix_png(confusion, output_path, title=f"{split_name.title()} Confusion Matrix")
        artifacts[f"{split_name}_confusion_matrix_png"] = str(output_path)

    loss_path = output_dir / "loss_by_epoch.png"
    save_metric_line_png(epochs, loss_path, metric_name="loss", title="Loss by Epoch", ylabel="Loss")
    artifacts["loss_by_epoch_png"] = str(loss_path)

    f1_path = output_dir / "f1_by_epoch.png"
    save_metric_line_png(epochs, f1_path, metric_name="f1", title="F1 by Epoch", ylabel="F1")
    artifacts["f1_by_epoch_png"] = str(f1_path)

    metrics_table = metrics_for_table(
        result_payload=result_payload,
        best_epoch_record=best_epoch_record,
        test_predictions=test_predictions,
        threshold=threshold,
    )
    best_epoch = best_epoch_record.get("epoch", "unknown")
    metrics_table_path = output_dir / "best_epoch_metrics_table.png"
    save_table_png(metrics_table, metrics_table_path, title=f"Best Epoch Metrics (epoch {best_epoch})")
    artifacts["best_epoch_metrics_table_png"] = str(metrics_table_path)

    sweep_table = build_threshold_sweep(val_predictions)
    sweep_csv_path = output_dir / "threshold_sweep.csv"
    save_threshold_sweep_csv(sweep_table, sweep_csv_path)
    artifacts["threshold_sweep_csv"] = str(sweep_csv_path)

    sweep_png_path = output_dir / "threshold_sweep_table.png"
    save_table_png(sweep_table, sweep_png_path, title="Validation Threshold Sweep", font_size=8)
    artifacts["threshold_sweep_table_png"] = str(sweep_png_path)

    return artifacts


def main() -> int:
    args = parse_args()
    try:
        artifacts = generate_visualizations(args.run_dir, args.output_dir)
    except (FileNotFoundError, VisualizationError, ValueError) as exc:
        print(f"Visualization generation failed: {exc}", file=sys.stderr)
        return 1

    print("CNN run visualization generation complete")
    print(f"run_dir={args.run_dir}")
    print(f"output_dir={args.output_dir}")
    print("artifacts=" + json.dumps(artifacts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
