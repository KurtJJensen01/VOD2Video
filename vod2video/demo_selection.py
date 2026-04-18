"""Demo example selection helpers for Branch 4C."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

from .prediction_review import (
    PredictionReviewError,
    load_prediction_review_csv,
    prepare_predictions_for_review,
)

DEFAULT_DEMO_OUTPUT_COLUMNS = (
    "demo_rank",
    "selection_group",
    "selection_reason",
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
    "threshold_margin",
    "distance_to_threshold",
)


class DemoSelectionError(ValueError):
    """Raised when demo selection inputs are invalid."""


@dataclass(frozen=True)
class DemoSelectionSummary:
    branch: str
    source_type: str
    source_prediction_csv: str
    source_review_csv: str | None
    source_model_summary_json: str | None
    source_experiment_name: str | None
    output_dir: str
    threshold: float
    top_k_per_group: int
    borderline_k: int
    total_rows: int
    labeled_rows: int
    review_group_counts: dict[str, int]
    selected_group_counts: dict[str, int]
    recommended_presentation_examples: dict[str, list[dict[str, object]]]
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_model_improvement_summary(path: Path) -> dict[str, object]:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model improvement summary not found: {resolved_path}")
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def resolve_prediction_csv_from_model_summary(path: Path) -> tuple[Path, dict[str, object]]:
    payload = load_model_improvement_summary(path)
    best_experiment = payload.get("best_experiment")
    if not isinstance(best_experiment, dict):
        raise DemoSelectionError("Model improvement summary is missing a best_experiment payload.")

    experiment_name = best_experiment.get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise DemoSelectionError("best_experiment is missing a valid experiment_name.")

    summary_dir = path.expanduser().resolve().parent
    prediction_csv = summary_dir / "runs" / experiment_name / "test_predictions.csv"
    if not prediction_csv.exists():
        raise FileNotFoundError(f"Best experiment prediction CSV not found: {prediction_csv}")
    return prediction_csv, payload


def prepare_demo_selection_frame(
    predictions_or_review: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    if predictions_or_review.empty:
        raise DemoSelectionError("Demo selection input contains zero rows.")

    if "predicted_probability" not in predictions_or_review.columns:
        raise DemoSelectionError("Input CSV must contain predicted_probability.")

    if "review_group" in predictions_or_review.columns and "score_rank" in predictions_or_review.columns:
        prepared = predictions_or_review.copy().reset_index(drop=True)
    else:
        prepared = prepare_predictions_for_review(predictions_or_review, threshold=threshold)

    prepared["predicted_probability"] = pd.to_numeric(prepared["predicted_probability"], errors="coerce")
    if prepared["predicted_probability"].isna().any():
        raise DemoSelectionError("predicted_probability must be numeric for every row.")

    prepared["threshold_margin"] = prepared["predicted_probability"] - float(threshold)
    prepared["distance_to_threshold"] = prepared["threshold_margin"].abs()
    prepared["presentation_priority"] = 0.0

    if "label" in prepared.columns:
        positive_mask = prepared["label"].fillna(-1).astype(float) == 1.0
        prepared.loc[positive_mask, "presentation_priority"] += 0.05

    review_bonus = {
        "true_positive": 0.15,
        "false_positive": 0.1,
        "false_negative": 0.08,
        "true_negative": 0.0,
        "unlabeled": 0.02,
    }
    if "review_group" in prepared.columns:
        prepared["presentation_priority"] += prepared["review_group"].map(review_bonus).fillna(0.0)

    prepared["presentation_priority"] += prepared["predicted_probability"]
    return prepared


def _select_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [column for column in DEFAULT_DEMO_OUTPUT_COLUMNS if column in frame.columns]
    remaining_columns = [column for column in frame.columns if column not in selected_columns]
    return frame.loc[:, [*selected_columns, *remaining_columns]]


def _finalize_selection(frame: pd.DataFrame, *, selection_group: str, selection_reason: str) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    output.insert(0, "demo_rank", range(1, len(output) + 1))
    output.insert(1, "selection_group", selection_group)
    output.insert(2, "selection_reason", selection_reason)
    return _select_output_columns(output)


def select_demo_candidates(
    prepared: pd.DataFrame,
    *,
    threshold: float,
    top_k_per_group: int,
    borderline_k: int,
) -> dict[str, pd.DataFrame]:
    top_k = max(int(top_k_per_group), 0)
    borderline_count = max(int(borderline_k), 0)

    ranked_highlights = prepared.sort_values(
        by=["predicted_probability", "presentation_priority", "score_rank"],
        ascending=[False, False, True],
        kind="mergesort",
    ).head(top_k)

    true_positives = prepared.loc[prepared["review_group"] == "true_positive"].sort_values(
        by=["predicted_probability", "score_rank"],
        ascending=[False, True],
        kind="mergesort",
    ).head(top_k)

    false_positives = prepared.loc[prepared["review_group"] == "false_positive"].sort_values(
        by=["predicted_probability", "score_rank"],
        ascending=[False, True],
        kind="mergesort",
    ).head(top_k)

    false_negatives = prepared.loc[prepared["review_group"] == "false_negative"].sort_values(
        by=["predicted_probability", "distance_to_threshold", "score_rank"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(top_k)

    borderline = prepared.sort_values(
        by=["distance_to_threshold", "predicted_probability", "score_rank"],
        ascending=[True, False, True],
        kind="mergesort",
    ).head(borderline_count)

    return {
        "top_ranked_highlights": _finalize_selection(
            ranked_highlights,
            selection_group="top_ranked_highlights",
            selection_reason="Highest predicted highlight probability overall.",
        ),
        "top_true_positives": _finalize_selection(
            true_positives,
            selection_group="top_true_positives",
            selection_reason="Highest-confidence correct highlight predictions.",
        ),
        "top_false_positives": _finalize_selection(
            false_positives,
            selection_group="top_false_positives",
            selection_reason="Highest-confidence highlight mistakes worth explaining.",
        ),
        "top_false_negatives": _finalize_selection(
            false_negatives,
            selection_group="top_false_negatives",
            selection_reason="Meaningful missed highlights closest to being recovered.",
        ),
        "borderline_examples": _finalize_selection(
            borderline,
            selection_group="borderline_examples",
            selection_reason="Examples closest to the decision threshold.",
        ),
    }


def build_demo_selection_summary(
    *,
    prepared: pd.DataFrame,
    selections: dict[str, pd.DataFrame],
    output_dir: Path,
    threshold: float,
    top_k_per_group: int,
    borderline_k: int,
    source_type: str,
    source_prediction_csv: Path,
    source_review_csv: Path | None,
    source_model_summary_json: Path | None,
    source_experiment_name: str | None,
) -> DemoSelectionSummary:
    review_group_counts = {
        str(group): int(count)
        for group, count in prepared["review_group"].value_counts(dropna=False).sort_index().items()
    }
    selected_group_counts = {
        name: int(len(frame))
        for name, frame in selections.items()
    }

    recommended: dict[str, list[dict[str, object]]] = {}
    for name in ("top_true_positives", "top_false_positives", "top_false_negatives", "top_ranked_highlights"):
        frame = selections.get(name)
        if frame is None or frame.empty:
            recommended[name] = []
            continue
        preview_columns = [
            column
            for column in (
                "demo_rank",
                "unique_id",
                "resolved_clip_path",
                "label",
                "predicted_probability",
                "predicted_class",
                "selection_reason",
            )
            if column in frame.columns
        ]
        recommended[name] = frame.loc[:, preview_columns].head(3).to_dict(orient="records")

    notes: list[str] = []
    if int(prepared["label"].notna().sum()) == 0:
        notes.append("Labels were unavailable, so TP/FP/FN group outputs may be empty.")
    if selections["top_false_negatives"].empty:
        notes.append("No labeled false negatives were available in the selected source predictions.")
    if selections["top_false_positives"].empty:
        notes.append("No labeled false positives were available in the selected source predictions.")

    return DemoSelectionSummary(
        branch="4c_demo_selection",
        source_type=source_type,
        source_prediction_csv=str(source_prediction_csv.expanduser().resolve()),
        source_review_csv=str(source_review_csv.expanduser().resolve()) if source_review_csv is not None else None,
        source_model_summary_json=(
            str(source_model_summary_json.expanduser().resolve())
            if source_model_summary_json is not None
            else None
        ),
        source_experiment_name=source_experiment_name,
        output_dir=str(output_dir.expanduser().resolve()),
        threshold=float(threshold),
        top_k_per_group=int(max(int(top_k_per_group), 0)),
        borderline_k=int(max(int(borderline_k), 0)),
        total_rows=int(len(prepared)),
        labeled_rows=int(prepared["label"].notna().sum()) if "label" in prepared.columns else 0,
        review_group_counts=review_group_counts,
        selected_group_counts=selected_group_counts,
        recommended_presentation_examples=recommended,
        notes=notes,
    )


def write_demo_selection_outputs(
    *,
    output_dir: Path,
    selections: dict[str, pd.DataFrame],
    summary: DemoSelectionSummary,
) -> dict[str, Path]:
    resolved_dir = output_dir.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary_json": resolved_dir / "demo_examples_summary.json",
        "top_true_positives_csv": resolved_dir / "top_true_positives.csv",
        "top_false_positives_csv": resolved_dir / "top_false_positives.csv",
        "top_false_negatives_csv": resolved_dir / "top_false_negatives.csv",
        "top_ranked_highlights_csv": resolved_dir / "top_ranked_highlights.csv",
        "borderline_examples_csv": resolved_dir / "borderline_examples.csv",
    }

    selections["top_true_positives"].to_csv(paths["top_true_positives_csv"], index=False)
    selections["top_false_positives"].to_csv(paths["top_false_positives_csv"], index=False)
    selections["top_false_negatives"].to_csv(paths["top_false_negatives_csv"], index=False)
    selections["top_ranked_highlights"].to_csv(paths["top_ranked_highlights_csv"], index=False)
    selections["borderline_examples"].to_csv(paths["borderline_examples_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return paths


def run_demo_selection(
    *,
    output_dir: Path,
    threshold: float,
    top_k_per_group: int,
    borderline_k: int,
    prediction_csv_path: Path | None = None,
    review_csv_path: Path | None = None,
    model_summary_json_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], DemoSelectionSummary, dict[str, Path]]:
    source_type = "prediction_csv"
    source_prediction_csv: Path | None = prediction_csv_path
    source_review_csv = review_csv_path
    source_model_summary_json = model_summary_json_path
    source_experiment_name: str | None = None

    if source_prediction_csv is None and source_review_csv is None:
        if model_summary_json_path is None:
            raise DemoSelectionError(
                "Provide --prediction-csv, --review-csv, or --model-summary-json for demo selection."
            )
        source_prediction_csv, model_summary = resolve_prediction_csv_from_model_summary(model_summary_json_path)
        source_type = "model_improvement_best_experiment"
        best_experiment = model_summary.get("best_experiment", {})
        if isinstance(best_experiment, dict):
            source_experiment_name = best_experiment.get("experiment_name")

    if source_review_csv is not None:
        raw_input = load_prediction_review_csv(source_review_csv)
        source_type = "review_csv" if source_prediction_csv is None else f"{source_type}+review_csv"
    elif source_prediction_csv is not None:
        raw_input = load_prediction_review_csv(source_prediction_csv)
    else:
        raise DemoSelectionError("Could not resolve an input CSV for demo selection.")

    try:
        prepared = prepare_demo_selection_frame(raw_input, threshold=threshold)
    except PredictionReviewError as exc:
        raise DemoSelectionError(str(exc)) from exc

    selections = select_demo_candidates(
        prepared,
        threshold=threshold,
        top_k_per_group=top_k_per_group,
        borderline_k=borderline_k,
    )
    summary = build_demo_selection_summary(
        prepared=prepared,
        selections=selections,
        output_dir=output_dir,
        threshold=threshold,
        top_k_per_group=top_k_per_group,
        borderline_k=borderline_k,
        source_type=source_type,
        source_prediction_csv=source_prediction_csv if source_prediction_csv is not None else source_review_csv,
        source_review_csv=source_review_csv,
        source_model_summary_json=source_model_summary_json,
        source_experiment_name=source_experiment_name,
    )
    output_paths = write_demo_selection_outputs(
        output_dir=output_dir,
        selections=selections,
        summary=summary,
    )
    return prepared, selections, summary, output_paths
