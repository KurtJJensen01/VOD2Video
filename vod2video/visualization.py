"""Result visualization helpers for Branch 3C."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

METRIC_COLUMNS = ("accuracy", "precision", "recall", "f1")
COUNT_COLUMNS = (
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "sample_count",
)


class VisualizationError(ValueError):
    """Raised when required visualization inputs are missing or malformed."""


@dataclass(frozen=True)
class VisualizationArtifacts:
    split_name: str
    output_dir: str
    metrics_table_csv: str
    confusion_matrix_csv: str
    visualization_summary_json: str
    metrics_comparison_csv: str | None = None
    review_summary_table_csv: str | None = None
    epoch_metrics_csv: str | None = None
    confusion_matrix_png: str | None = None
    metrics_table_png: str | None = None
    review_summary_table_png: str | None = None
    metric_bar_chart_png: str | None = None
    epoch_overview_png: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"JSON file not found: {resolved_path}")
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def _optional_load_json(path: Path) -> dict[str, Any] | None:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        return None
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def _resolve_review_summary_path(review_dir: Path | None, split_name: str) -> Path | None:
    if review_dir is None:
        return None

    resolved_dir = review_dir.expanduser().resolve()
    if resolved_dir.is_file():
        return resolved_dir

    candidates = [
        resolved_dir / "review_summary.json",
        resolved_dir / f"{split_name}_review" / "review_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _coerce_metric_row(name: str, metrics: dict[str, Any], *, split_name: str | None = None) -> dict[str, Any]:
    row = {
        "run_name": name,
        "split": split_name,
    }
    for column in METRIC_COLUMNS:
        row[column] = metrics.get(column)
    for column in COUNT_COLUMNS:
        row[column] = metrics.get(column)
    return row


def build_confusion_matrix_table(confusion_matrix: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "actual_class": "non_highlight",
                "predicted_non_highlight": int(confusion_matrix["true_negative"]),
                "predicted_highlight": int(confusion_matrix["false_positive"]),
            },
            {
                "actual_class": "highlight",
                "predicted_non_highlight": int(confusion_matrix["false_negative"]),
                "predicted_highlight": int(confusion_matrix["true_positive"]),
            },
        ]
    )


def build_metrics_table(metrics_summary: dict[str, Any]) -> pd.DataFrame:
    best_epoch_metrics = metrics_summary.get("best_epoch_metrics")
    if not isinstance(best_epoch_metrics, dict):
        raise VisualizationError("metrics_summary.json is missing best_epoch_metrics.")

    rows = []
    for split_name in ("train", "val", "test"):
        split_metrics = best_epoch_metrics.get(split_name)
        if isinstance(split_metrics, dict):
            row = _coerce_metric_row("real_feature_baseline", split_metrics, split_name=split_name)
            row["epoch"] = int(metrics_summary.get("best_epoch", 0))
            rows.append(row)
    if not rows:
        raise VisualizationError("No split metrics were found in metrics_summary.json.")
    return pd.DataFrame(rows)


def build_metrics_comparison_table(
    evaluation_summary: dict[str, Any],
    *,
    split_name: str,
    comparison_summary: dict[str, Any] | None = None,
) -> pd.DataFrame:
    split_payload = evaluation_summary.get(split_name)
    if not isinstance(split_payload, dict):
        raise VisualizationError(f"evaluation_summary.json is missing the '{split_name}' split.")
    if not isinstance(split_payload.get("metrics"), dict):
        raise VisualizationError(f"evaluation_summary.json split '{split_name}' is missing metrics.")

    rows = [
        _coerce_metric_row(
            "real_feature_baseline",
            split_payload["metrics"],
            split_name=split_name,
        )
    ]
    if (
        split_name == "test"
        and comparison_summary is not None
        and isinstance(comparison_summary.get("test_metrics"), dict)
    ):
        rows.append(
            _coerce_metric_row(
                str(comparison_summary.get("name", "comparison_baseline")),
                comparison_summary["test_metrics"],
                split_name="test",
            )
        )
    return pd.DataFrame(rows)


def build_epoch_metrics_table(training_history: dict[str, Any]) -> pd.DataFrame:
    epochs = training_history.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise VisualizationError("training_history.json does not contain any epoch records.")

    rows: list[dict[str, Any]] = []
    for record in epochs:
        epoch = int(record["epoch"])
        train_metrics = record.get("train", {})
        val_metrics = record.get("val", {})
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss"),
                "val_loss": val_metrics.get("loss"),
                "train_accuracy": train_metrics.get("accuracy"),
                "val_accuracy": val_metrics.get("accuracy"),
                "train_precision": train_metrics.get("precision"),
                "val_precision": val_metrics.get("precision"),
                "train_recall": train_metrics.get("recall"),
                "val_recall": val_metrics.get("recall"),
                "train_f1": train_metrics.get("f1"),
                "val_f1": val_metrics.get("f1"),
            }
        )
    return pd.DataFrame(rows)


def build_review_summary_table(review_summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    review_group_counts = review_summary.get("review_group_counts", {})
    for group_name, count in review_group_counts.items():
        rows.append(
            {
                "section": "review_group_counts",
                "name": group_name,
                "value": int(count),
            }
        )

    top_k_summary = review_summary.get("top_k_summary", {})
    for metric_name, value in top_k_summary.items():
        rows.append(
            {
                "section": "top_k_summary",
                "name": metric_name,
                "value": value,
            }
        )

    if not rows:
        raise VisualizationError("review_summary.json does not contain review_group_counts or top_k_summary.")
    return pd.DataFrame(rows)


def _prepare_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def render_confusion_matrix_png(confusion_table: pd.DataFrame, output_path: Path, *, title: str) -> Path:
    plt = _prepare_matplotlib()
    values = confusion_table.loc[:, ["predicted_non_highlight", "predicted_highlight"]].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    ax.set_title(title)

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            ax.text(
                column_index,
                row_index,
                str(int(values[row_index, column_index])),
                ha="center",
                va="center",
                color="black",
                fontsize=12,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_metric_bar_chart_png(comparison_table: pd.DataFrame, output_path: Path, *, title: str) -> Path:
    plt = _prepare_matplotlib()

    chart_frame = comparison_table.loc[:, ["run_name", *METRIC_COLUMNS]].copy()
    chart_frame = chart_frame.set_index("run_name")

    fig, ax = plt.subplots(figsize=(8, 5))
    chart_frame.plot(kind="bar", ax=ax, width=0.75, color=["#4E79A7", "#F28E2B", "#59A14F", "#E15759"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_epoch_overview_png(epoch_table: pd.DataFrame, output_path: Path, *, title: str, best_epoch: int) -> Path:
    plt = _prepare_matplotlib()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5))

    axes[0].plot(epoch_table["epoch"], epoch_table["train_loss"], marker="o", label="Train loss", color="#4E79A7")
    axes[0].plot(epoch_table["epoch"], epoch_table["val_loss"], marker="o", label="Val loss", color="#F28E2B")
    axes[0].axvline(best_epoch, color="#7F7F7F", linestyle="--", linewidth=1)
    axes[0].set_title("Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epoch_table["epoch"], epoch_table["train_f1"], marker="o", label="Train F1", color="#59A14F")
    axes[1].plot(epoch_table["epoch"], epoch_table["val_f1"], marker="o", label="Val F1", color="#E15759")
    axes[1].axvline(best_epoch, color="#7F7F7F", linestyle="--", linewidth=1)
    axes[1].set_title("F1 by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_table_png(dataframe: pd.DataFrame, output_path: Path, *, title: str) -> Path:
    plt = _prepare_matplotlib()

    display_frame = dataframe.copy()
    for column in display_frame.columns:
        if pd.api.types.is_float_dtype(display_frame[column]):
            display_frame[column] = display_frame[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )

    row_count = max(len(display_frame), 1)
    fig_height = max(2.5, 1.1 + row_count * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_frame.values,
        colLabels=display_frame.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_visualization_artifacts(
    *,
    training_dir: Path,
    output_dir: Path,
    split_name: str = "test",
    review_dir: Path | None = None,
    generate_plots: bool = True,
) -> VisualizationArtifacts:
    resolved_training_dir = training_dir.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary = _load_json(resolved_training_dir / "metrics_summary.json")
    evaluation_summary = _load_json(resolved_training_dir / "evaluation_summary.json")
    training_history = _load_json(resolved_training_dir / "training_history.json")
    comparison_summary = _optional_load_json(resolved_training_dir / "comparison_summary.json")

    review_summary_path = _resolve_review_summary_path(review_dir, split_name)
    review_summary = _load_json(review_summary_path) if review_summary_path is not None else None

    split_payload = evaluation_summary.get(split_name)
    if not isinstance(split_payload, dict):
        raise VisualizationError(f"evaluation_summary.json is missing the '{split_name}' split.")

    confusion_table = build_confusion_matrix_table(split_payload["confusion_matrix"])
    metrics_table = build_metrics_table(metrics_summary)
    comparison_table = build_metrics_comparison_table(
        evaluation_summary,
        split_name=split_name,
        comparison_summary=comparison_summary,
    )
    epoch_table = build_epoch_metrics_table(training_history)

    confusion_csv_path = resolved_output_dir / f"{split_name}_confusion_matrix.csv"
    confusion_json_path = resolved_output_dir / f"{split_name}_confusion_matrix.json"
    metrics_csv_path = resolved_output_dir / "metrics_table.csv"
    comparison_csv_path = resolved_output_dir / "metrics_comparison.csv"
    epoch_csv_path = resolved_output_dir / "epoch_metrics.csv"

    confusion_table.to_csv(confusion_csv_path, index=False)
    confusion_json_path.write_text(
        json.dumps(split_payload["confusion_matrix"], indent=2),
        encoding="utf-8",
    )
    metrics_table.to_csv(metrics_csv_path, index=False)
    comparison_table.to_csv(comparison_csv_path, index=False)
    epoch_table.to_csv(epoch_csv_path, index=False)

    review_table_path: Path | None = None
    top_k_table_path: Path | None = None
    if review_summary is not None:
        review_table = build_review_summary_table(review_summary)
        review_table_path = resolved_output_dir / f"{split_name}_review_summary_table.csv"
        review_table.to_csv(review_table_path, index=False)

        top_k_table_path = resolved_output_dir / f"{split_name}_top_k_summary.json"
        top_k_table_path.write_text(
            json.dumps(review_summary.get("top_k_summary", {}), indent=2),
            encoding="utf-8",
        )

    confusion_png_path: Path | None = None
    metrics_png_path: Path | None = None
    metric_chart_path: Path | None = None
    epoch_png_path: Path | None = None
    review_png_path: Path | None = None

    if generate_plots:
        try:
            confusion_png_path = render_confusion_matrix_png(
                confusion_table,
                resolved_output_dir / f"{split_name}_confusion_matrix.png",
                title=f"{split_name.title()} Confusion Matrix",
            )
            metrics_png_path = render_table_png(
                metrics_table.loc[:, ["split", "accuracy", "precision", "recall", "f1"]],
                resolved_output_dir / "metrics_table.png",
                title="Best-Epoch Metrics by Split",
            )
            metric_chart_path = render_metric_bar_chart_png(
                comparison_table.loc[:, ["run_name", *METRIC_COLUMNS]],
                resolved_output_dir / f"{split_name}_metric_bar_chart.png",
                title=f"{split_name.title()} Metric Comparison",
            )
            epoch_png_path = render_epoch_overview_png(
                epoch_table,
                resolved_output_dir / "epoch_overview.png",
                title="Training Overview",
                best_epoch=int(metrics_summary["best_epoch"]),
            )
            if review_summary is not None:
                review_png_path = render_table_png(
                    review_table,
                    resolved_output_dir / f"{split_name}_review_summary_table.png",
                    title=f"{split_name.title()} Review Summary",
                )
        except ModuleNotFoundError:
            confusion_png_path = None
            metrics_png_path = None
            metric_chart_path = None
            epoch_png_path = None
            review_png_path = None

    best_presentation_artifacts = [
        str(confusion_png_path or confusion_csv_path),
        str(metrics_png_path or metrics_csv_path),
        str(metric_chart_path or comparison_csv_path),
    ]
    if epoch_png_path is not None:
        best_presentation_artifacts.append(str(epoch_png_path))
    if review_table_path is not None:
        best_presentation_artifacts.append(str(review_png_path or review_table_path))

    summary_payload = {
        "branch": "3c_result_visualization",
        "split_name": split_name,
        "training_dir": str(resolved_training_dir),
        "review_summary_path": str(review_summary_path) if review_summary_path is not None else None,
        "output_dir": str(resolved_output_dir),
        "best_epoch": int(metrics_summary["best_epoch"]),
        "best_metric_value": metrics_summary.get("best_metric_value"),
        "generated_files": {
            "confusion_matrix_csv": str(confusion_csv_path),
            "confusion_matrix_json": str(confusion_json_path),
            "metrics_table_csv": str(metrics_csv_path),
            "metrics_comparison_csv": str(comparison_csv_path),
            "epoch_metrics_csv": str(epoch_csv_path),
            "review_summary_table_csv": str(review_table_path) if review_table_path is not None else None,
            "top_k_summary_json": str(top_k_table_path) if top_k_table_path is not None else None,
            "confusion_matrix_png": str(confusion_png_path) if confusion_png_path is not None else None,
            "metrics_table_png": str(metrics_png_path) if metrics_png_path is not None else None,
            "review_summary_table_png": str(review_png_path) if review_png_path is not None else None,
            "metric_bar_chart_png": str(metric_chart_path) if metric_chart_path is not None else None,
            "epoch_overview_png": str(epoch_png_path) if epoch_png_path is not None else None,
        },
        "presentation_recommendations": {
            "best_slide_artifacts": best_presentation_artifacts,
            "headline_metrics": comparison_table.loc[:, ["run_name", "accuracy", "precision", "recall", "f1"]]
            .to_dict(orient="records"),
            "error_distribution": split_payload["confusion_matrix"],
            "review_group_counts": review_summary.get("review_group_counts") if review_summary is not None else None,
        },
    }

    summary_json_path = resolved_output_dir / "visualization_summary.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return VisualizationArtifacts(
        split_name=split_name,
        output_dir=str(resolved_output_dir),
        metrics_table_csv=str(metrics_csv_path),
        confusion_matrix_csv=str(confusion_csv_path),
        visualization_summary_json=str(summary_json_path),
        metrics_comparison_csv=str(comparison_csv_path),
        review_summary_table_csv=str(review_table_path) if review_table_path is not None else None,
        epoch_metrics_csv=str(epoch_csv_path),
        confusion_matrix_png=str(confusion_png_path) if confusion_png_path is not None else None,
        metrics_table_png=str(metrics_png_path) if metrics_png_path is not None else None,
        review_summary_table_png=str(review_png_path) if review_png_path is not None else None,
        metric_bar_chart_png=str(metric_chart_path) if metric_chart_path is not None else None,
        epoch_overview_png=str(epoch_png_path) if epoch_png_path is not None else None,
    )
