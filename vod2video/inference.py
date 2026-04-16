"""Inference and demo-output helpers for Branch 2C."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .models import build_model
from .training_config import ModelConfig

DEFAULT_OUTPUT_COLUMNS = (
    "score_rank",
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


class InferenceError(ValueError):
    """Raised when inference inputs cannot be prepared or scored."""


@dataclass(frozen=True)
class LoadedInferenceCheckpoint:
    checkpoint_path: str
    model: torch.nn.Module
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    feature_names: tuple[str, ...]
    normalization_stats: dict[str, dict[str, float]] | None
    threshold: float
    device: str
    epoch: int | None
    metrics: dict[str, float | int] | None


@dataclass(frozen=True)
class InferenceSummary:
    checkpoint_path: str
    feature_manifest_path: str
    scored_rows: int
    predicted_positive_count: int
    threshold: float
    device: str
    normalization_applied: bool
    normalization_source: str
    score_min: float
    score_max: float
    score_mean: float
    feature_names: tuple[str, ...]
    top_examples: list[dict[str, object]]


class FeatureManifestInferenceDataset(Dataset):
    """Tensor-backed dataset for manifest-only inference."""

    def __init__(self, features: torch.Tensor) -> None:
        self.features = features

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[index]}


def load_inference_checkpoint(
    checkpoint_path: Path,
    *,
    device: str | None = None,
    threshold: float | None = None,
) -> LoadedInferenceCheckpoint:
    resolved_path = checkpoint_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")

    requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(resolved_path, map_location=requested_device)

    model_config_data = payload.get("model_config")
    if not isinstance(model_config_data, dict):
        raise InferenceError("Checkpoint is missing a valid model_config payload.")

    feature_names = payload.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise InferenceError("Checkpoint is missing a non-empty feature_names list.")

    model_config = ModelConfig(**model_config_data)
    if model_config.input_dim != len(feature_names):
        raise InferenceError(
            f"Checkpoint input_dim={model_config.input_dim} does not match "
            f"{len(feature_names)} feature columns."
        )

    model = build_model(model_config)
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(torch.device(requested_device))
    model.eval()

    training_config = payload.get("training_config")
    if not isinstance(training_config, dict):
        training_config = {}

    saved_threshold = training_config.get("decision_threshold", 0.5)
    active_threshold = float(threshold if threshold is not None else saved_threshold)

    normalization_stats = payload.get("normalization_stats")
    if normalization_stats is not None and not isinstance(normalization_stats, dict):
        raise InferenceError("Checkpoint normalization_stats must be a dict when present.")

    metrics = payload.get("metrics")
    if metrics is not None and not isinstance(metrics, dict):
        metrics = None

    return LoadedInferenceCheckpoint(
        checkpoint_path=str(resolved_path),
        model=model,
        model_config=model_config_data,
        training_config=training_config,
        feature_names=tuple(str(name) for name in feature_names),
        normalization_stats=normalization_stats,
        threshold=active_threshold,
        device=requested_device,
        epoch=payload.get("epoch"),
        metrics=metrics,
    )


def load_feature_manifest_for_inference(path: Path) -> pd.DataFrame:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {resolved_path}")
    return pd.read_csv(resolved_path)


def prepare_feature_frame_for_inference(
    dataframe: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    normalization_stats: dict[str, dict[str, float]] | None,
) -> tuple[pd.DataFrame, bool, str]:
    missing_columns = [name for name in feature_names if name not in dataframe.columns]
    if missing_columns:
        raise InferenceError(
            f"Feature manifest is missing required feature column(s): {', '.join(missing_columns)}"
        )

    feature_frame = dataframe.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    if feature_frame.isna().any().any():
        bad_columns = feature_frame.columns[feature_frame.isna().any()].tolist()
        raise InferenceError(
            f"Feature manifest contains missing or non-numeric values: {', '.join(bad_columns)}"
        )

    if normalization_stats is None:
        derived_stats = _derive_manifest_normalization_stats(dataframe, feature_names=feature_names)
        if derived_stats is None:
            return feature_frame.astype("float32"), False, "not_applied"
        normalization_stats = derived_stats
        normalization_source = "manifest_train_split" if "split" in dataframe.columns and (dataframe["split"] == "train").any() else "manifest_all_rows"
    else:
        normalization_source = "checkpoint"

    means = normalization_stats.get("means", {})
    stds = normalization_stats.get("stds", {})
    missing_stats = [name for name in feature_names if name not in means or name not in stds]
    if missing_stats:
        raise InferenceError(
            f"Checkpoint normalization_stats is missing feature(s): {', '.join(missing_stats)}"
        )

    normalized = feature_frame.copy()
    for name in feature_names:
        std = float(stds[name])
        if std == 0.0:
            std = 1.0
        normalized[name] = (feature_frame[name] - float(means[name])) / std

    return normalized.astype("float32"), True, normalization_source


def _derive_manifest_normalization_stats(
    dataframe: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
) -> dict[str, dict[str, float]] | None:
    if dataframe.empty:
        return None

    if "split" in dataframe.columns:
        train_rows = dataframe.loc[dataframe["split"] == "train"].copy()
        if not train_rows.empty:
            source_frame = train_rows
        else:
            source_frame = dataframe
    else:
        source_frame = dataframe

    feature_frame = source_frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    if feature_frame.isna().any().any():
        return None

    feature_means = feature_frame.mean()
    feature_stds = feature_frame.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return {
        "means": {column: float(value) for column, value in feature_means.items()},
        "stds": {column: float(value) for column, value in feature_stds.items()},
    }


def _run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    device: str,
) -> torch.Tensor:
    collected_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            logits = model(features)
            collected_logits.append(logits.detach().cpu())

    if not collected_logits:
        raise InferenceError("Feature manifest produced zero rows to score.")
    return torch.cat(collected_logits)


def build_prediction_dataframe(
    manifest_dataframe: pd.DataFrame,
    probabilities: torch.Tensor,
    *,
    threshold: float,
) -> pd.DataFrame:
    if len(manifest_dataframe) != int(probabilities.numel()):
        raise InferenceError("Prediction count does not match feature manifest row count.")

    output = manifest_dataframe.copy().reset_index(drop=True)
    output["predicted_probability"] = probabilities.numpy().astype("float32")
    output["predicted_class"] = (output["predicted_probability"] >= threshold).astype(int)
    output = output.sort_values(
        by=["predicted_probability", "unique_id"] if "unique_id" in output.columns else ["predicted_probability"],
        ascending=[False, True] if "unique_id" in output.columns else [False],
        kind="mergesort",
    ).reset_index(drop=True)
    output["score_rank"] = range(1, len(output) + 1)

    selected_columns = [column for column in DEFAULT_OUTPUT_COLUMNS if column in output.columns]
    remaining_columns = [column for column in output.columns if column not in selected_columns]
    return output.loc[:, [*selected_columns, *remaining_columns]]


def score_feature_manifest(
    *,
    checkpoint_path: Path,
    feature_manifest_path: Path,
    output_dir: Path | None = None,
    batch_size: int = 64,
    threshold: float | None = None,
    top_k: int = 10,
    device: str | None = None,
) -> tuple[pd.DataFrame, InferenceSummary, dict[str, Path] | None]:
    checkpoint = load_inference_checkpoint(
        checkpoint_path,
        device=device,
        threshold=threshold,
    )
    feature_manifest = load_feature_manifest_for_inference(feature_manifest_path)
    prepared_features, normalization_applied, normalization_source = prepare_feature_frame_for_inference(
        feature_manifest,
        feature_names=checkpoint.feature_names,
        normalization_stats=checkpoint.normalization_stats,
    )

    dataset = FeatureManifestInferenceDataset(
        torch.tensor(prepared_features.to_numpy(dtype="float32"), dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits = _run_inference(checkpoint.model, dataloader, device=checkpoint.device)
    probabilities = torch.sigmoid(logits)

    predictions = build_prediction_dataframe(
        feature_manifest,
        probabilities,
        threshold=checkpoint.threshold,
    )

    top_rows = predictions.head(max(int(top_k), 0))
    summary = InferenceSummary(
        checkpoint_path=checkpoint.checkpoint_path,
        feature_manifest_path=str(feature_manifest_path.expanduser().resolve()),
        scored_rows=int(len(predictions)),
        predicted_positive_count=int(predictions["predicted_class"].sum()),
        threshold=float(checkpoint.threshold),
        device=checkpoint.device,
        normalization_applied=normalization_applied,
        normalization_source=normalization_source,
        score_min=float(predictions["predicted_probability"].min()),
        score_max=float(predictions["predicted_probability"].max()),
        score_mean=float(predictions["predicted_probability"].mean()),
        feature_names=checkpoint.feature_names,
        top_examples=top_rows.loc[
            :,
            [column for column in ("score_rank", "unique_id", "vod_id", "segment_id", "predicted_probability") if column in top_rows.columns],
        ].to_dict(orient="records"),
    )

    output_paths = None
    if output_dir is not None:
        output_paths = write_inference_outputs(
            predictions,
            output_dir=output_dir,
            summary=summary,
            top_k=top_k,
        )

    return predictions, summary, output_paths


def write_inference_outputs(
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    summary: InferenceSummary,
    top_k: int,
) -> dict[str, Path]:
    resolved_dir = output_dir.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)

    top_count = max(int(top_k), 0)
    top_predictions = predictions.head(top_count).copy()

    paths = {
        "scored_csv": resolved_dir / "scored_clips.csv",
        "top_csv": resolved_dir / "top_highlights.csv",
        "summary_json": resolved_dir / "inference_summary.json",
    }
    predictions.to_csv(paths["scored_csv"], index=False)
    top_predictions.to_csv(paths["top_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return paths
