"""Prediction review helpers for Branch 3B."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

DEFAULT_REVIEW_OUTPUT_COLUMNS = (
    "score_rank",
    "review_group",
    "prediction_correct",
    "unique_id",
    "vod_id",
    "segment_id",
    "clip_path",
    "resolved_clip_path",
    "split",
    "label",
    "predicted_probability",
    "predicted_class",
)


class PredictionReviewError(ValueError):
    """Raised when prediction review inputs are invalid."""


@dataclass(frozen=True)
class PredictionReviewSummary:
    prediction_csv_path: str
    output_dir: str | None
    threshold: float
    top_k: int
    total_rows: int
    labeled_rows: int
    unlabeled_rows: int
    labels_available: bool
    label_join_keys: list[str]
    review_group_counts: dict[str, int]
    predicted_class_counts: dict[str, int]
    top_k_summary: dict[str, int | float | None]
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_prediction_review_csv(path: Path) -> pd.DataFrame:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {resolved_path}")
    return pd.read_csv(resolved_path)


def _normalize_label_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid_mask = numeric.isin([0, 1]) | numeric.isna()
    if not valid_mask.all():
        raise PredictionReviewError("Label column must contain only 0/1 values when present.")
    return numeric.astype("Int64")


def _resolve_join_keys(predictions: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    candidate_keys = [
        ["unique_id"],
        ["vod_id", "segment_id"],
        ["resolved_clip_path"],
        ["clip_path"],
    ]
    for keys in candidate_keys:
        if all(key in predictions.columns and key in labels.columns for key in keys):
            return keys
    raise PredictionReviewError(
        "Could not find shared join keys between predictions and labels. "
        "Expected one of: unique_id, (vod_id + segment_id), resolved_clip_path, clip_path."
    )


def merge_review_labels(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    join_keys = _resolve_join_keys(predictions, labels)
    if "label" not in labels.columns:
        raise PredictionReviewError("Labels CSV must contain a label column.")

    label_frame = labels.loc[:, [*join_keys, "label"]].copy()
    label_frame["label"] = _normalize_label_series(label_frame["label"])
    label_frame = label_frame.dropna(subset=join_keys)
    label_frame = label_frame.drop_duplicates(subset=join_keys, keep="last")

    base = predictions.copy()
    existing_label = None
    if "label" in base.columns:
        existing_label = _normalize_label_series(base["label"])
        base = base.drop(columns=["label"])

    merged = base.merge(
        label_frame.rename(columns={"label": "merged_label"}),
        on=join_keys,
        how="left",
    )
    if existing_label is not None:
        merged["label"] = existing_label.combine_first(merged["merged_label"])
    else:
        merged["label"] = merged["merged_label"]
    merged = merged.drop(columns=["merged_label"])
    return merged, join_keys


def prepare_predictions_for_review(
    predictions: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    if predictions.empty:
        raise PredictionReviewError("Prediction CSV contains zero rows.")
    if "predicted_probability" not in predictions.columns:
        raise PredictionReviewError("Prediction CSV must contain a predicted_probability column.")

    prepared = predictions.copy().reset_index(drop=True)
    prepared["predicted_probability"] = pd.to_numeric(prepared["predicted_probability"], errors="coerce")
    if prepared["predicted_probability"].isna().any():
        raise PredictionReviewError("predicted_probability must be numeric for every row.")

    if "predicted_class" in prepared.columns:
        prepared["predicted_class"] = pd.to_numeric(prepared["predicted_class"], errors="coerce")
        if prepared["predicted_class"].isna().any():
            raise PredictionReviewError("predicted_class must be numeric when present.")
        prepared["predicted_class"] = prepared["predicted_class"].astype(int)
    else:
        prepared["predicted_class"] = (prepared["predicted_probability"] >= float(threshold)).astype(int)

    if "label" in prepared.columns:
        prepared["label"] = _normalize_label_series(prepared["label"])
    else:
        prepared["label"] = pd.Series(pd.array([pd.NA] * len(prepared), dtype="Int64"))

    sort_columns = ["predicted_probability"]
    ascending = [False]
    if "unique_id" in prepared.columns:
        sort_columns.append("unique_id")
        ascending.append(True)
    prepared = prepared.sort_values(by=sort_columns, ascending=ascending, kind="mergesort").reset_index(drop=True)
    prepared["score_rank"] = range(1, len(prepared) + 1)

    prepared["prediction_correct"] = pd.Series(pd.array([pd.NA] * len(prepared), dtype="boolean"))
    prepared["review_group"] = "unlabeled"

    labeled_mask = prepared["label"].notna()
    if labeled_mask.any():
        prepared.loc[labeled_mask, "prediction_correct"] = (
            prepared.loc[labeled_mask, "label"].astype(int) == prepared.loc[labeled_mask, "predicted_class"]
        )

        true_positive_mask = labeled_mask & (prepared["label"] == 1) & (prepared["predicted_class"] == 1)
        false_positive_mask = labeled_mask & (prepared["label"] == 0) & (prepared["predicted_class"] == 1)
        false_negative_mask = labeled_mask & (prepared["label"] == 1) & (prepared["predicted_class"] == 0)
        true_negative_mask = labeled_mask & (prepared["label"] == 0) & (prepared["predicted_class"] == 0)

        prepared.loc[true_positive_mask, "review_group"] = "true_positive"
        prepared.loc[false_positive_mask, "review_group"] = "false_positive"
        prepared.loc[false_negative_mask, "review_group"] = "false_negative"
        prepared.loc[true_negative_mask, "review_group"] = "true_negative"

    selected_columns = [column for column in DEFAULT_REVIEW_OUTPUT_COLUMNS if column in prepared.columns]
    remaining_columns = [column for column in prepared.columns if column not in selected_columns]
    return prepared.loc[:, [*selected_columns, *remaining_columns]]


def _build_top_k_summary(prepared: pd.DataFrame, *, top_k: int) -> dict[str, int | float | None]:
    top_frame = prepared.head(max(int(top_k), 0)).copy()
    summary: dict[str, int | float | None] = {
        "top_k": int(max(int(top_k), 0)),
        "top_k_rows": int(len(top_frame)),
        "top_k_predicted_positive_count": int((top_frame["predicted_class"] == 1).sum()),
        "top_k_probability_mean": float(top_frame["predicted_probability"].mean()) if not top_frame.empty else None,
    }

    labeled_top = top_frame.loc[top_frame["label"].notna()].copy()
    if not labeled_top.empty:
        summary["top_k_labeled_rows"] = int(len(labeled_top))
        summary["top_k_correct_count"] = int((labeled_top["label"].astype(int) == labeled_top["predicted_class"]).sum())
        summary["top_k_true_highlight_count"] = int((labeled_top["label"].astype(int) == 1).sum())
    return summary


def build_prediction_review_summary(
    prepared: pd.DataFrame,
    *,
    prediction_csv_path: Path,
    output_dir: Path | None,
    threshold: float,
    top_k: int,
    label_join_keys: list[str] | None = None,
) -> PredictionReviewSummary:
    labeled_rows = int(prepared["label"].notna().sum())
    predicted_counts = {
        str(label): int(count)
        for label, count in prepared["predicted_class"].value_counts(dropna=False).sort_index().items()
    }
    review_counts = {
        str(group): int(count)
        for group, count in prepared["review_group"].value_counts(dropna=False).sort_index().items()
    }

    notes: list[str] = []
    if labeled_rows == 0:
        notes.append("Label-based review is unavailable because no labels were found.")
    elif labeled_rows < len(prepared):
        notes.append("Only labeled rows were used for TP/FP/FN/TN review counts.")

    return PredictionReviewSummary(
        prediction_csv_path=str(prediction_csv_path.expanduser().resolve()),
        output_dir=str(output_dir.expanduser().resolve()) if output_dir is not None else None,
        threshold=float(threshold),
        top_k=int(max(int(top_k), 0)),
        total_rows=int(len(prepared)),
        labeled_rows=labeled_rows,
        unlabeled_rows=int(len(prepared) - labeled_rows),
        labels_available=bool(labeled_rows > 0),
        label_join_keys=list(label_join_keys or []),
        review_group_counts=review_counts,
        predicted_class_counts=predicted_counts,
        top_k_summary=_build_top_k_summary(prepared, top_k=top_k),
        notes=notes,
    )


def _select_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [column for column in DEFAULT_REVIEW_OUTPUT_COLUMNS if column in frame.columns]
    remaining_columns = [column for column in frame.columns if column not in selected_columns]
    return frame.loc[:, [*selected_columns, *remaining_columns]]


def write_prediction_review_outputs(
    prepared: pd.DataFrame,
    *,
    output_dir: Path,
    summary: PredictionReviewSummary,
) -> dict[str, Path]:
    resolved_dir = output_dir.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)

    top_k = int(summary.top_k)
    top_highlights = prepared.head(top_k).copy()
    top_non_highlights = prepared.sort_values(
        by=["predicted_probability", "score_rank"],
        ascending=[True, True],
        kind="mergesort",
    ).head(top_k)

    paths = {
        "ranked_predictions_csv": resolved_dir / "ranked_predictions.csv",
        "top_predicted_highlights_csv": resolved_dir / "top_predicted_highlights.csv",
        "top_predicted_non_highlights_csv": resolved_dir / "top_predicted_non_highlights.csv",
        "review_summary_json": resolved_dir / "review_summary.json",
    }
    _select_output_columns(prepared).to_csv(paths["ranked_predictions_csv"], index=False)
    _select_output_columns(top_highlights).to_csv(paths["top_predicted_highlights_csv"], index=False)
    _select_output_columns(top_non_highlights).to_csv(paths["top_predicted_non_highlights_csv"], index=False)

    labeled_frame = prepared.loc[prepared["label"].notna()].copy()
    if not labeled_frame.empty:
        review_groups = {
            "true_positive": "true_positives.csv",
            "false_positive": "false_positives.csv",
            "false_negative": "false_negatives.csv",
            "true_negative": "true_negatives.csv",
        }
        for review_group, filename in review_groups.items():
            group_frame = labeled_frame.loc[labeled_frame["review_group"] == review_group].copy()
            path = resolved_dir / filename
            _select_output_columns(group_frame).to_csv(path, index=False)
            paths[f"{review_group}_csv"] = path

    paths["review_summary_json"].write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return paths


def review_prediction_csv(
    *,
    prediction_csv_path: Path,
    output_dir: Path | None = None,
    labels_csv_path: Path | None = None,
    threshold: float = 0.5,
    top_k: int = 10,
) -> tuple[pd.DataFrame, PredictionReviewSummary, dict[str, Path] | None]:
    predictions = load_prediction_review_csv(prediction_csv_path)
    label_join_keys: list[str] = []
    if labels_csv_path is not None:
        labels = load_prediction_review_csv(labels_csv_path)
        predictions, label_join_keys = merge_review_labels(predictions, labels)

    prepared = prepare_predictions_for_review(predictions, threshold=threshold)
    summary = build_prediction_review_summary(
        prepared,
        prediction_csv_path=prediction_csv_path,
        output_dir=output_dir,
        threshold=threshold,
        top_k=top_k,
        label_join_keys=label_join_keys,
    )

    output_paths = None
    if output_dir is not None:
        output_paths = write_prediction_review_outputs(
            prepared,
            output_dir=output_dir,
            summary=summary,
        )
    return prepared, summary, output_paths
