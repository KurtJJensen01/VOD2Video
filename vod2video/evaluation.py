"""Checkpoint evaluation helpers for labeled manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .inference import (
    InferenceError,
    build_prediction_dataframe,
    load_feature_manifest_for_inference,
    load_inference_checkpoint,
    prepare_feature_frame_for_inference,
)
from .metrics import BinaryClassificationMetrics, compute_binary_classification_metrics
from .training_data import VideoAudioSplitManifestDataset, prepare_training_manifest


@dataclass(frozen=True)
class EvaluationArtifacts:
    split_name: str
    metrics: BinaryClassificationMetrics
    threshold: float
    checkpoint_path: str
    manifest_path: str
    normalization_applied: bool
    normalization_source: str

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        data["confusion_matrix"] = {
            "true_negative": self.metrics.true_negatives,
            "false_positive": self.metrics.false_positives,
            "false_negative": self.metrics.false_negatives,
            "true_positive": self.metrics.true_positives,
        }
        return data


class _FeatureOnlyDataset(Dataset):
    def __init__(self, features: torch.Tensor) -> None:
        self.features = features

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[index]}


def evaluate_checkpoint_on_manifest(
    *,
    checkpoint_path: Path,
    feature_manifest_path: Path,
    split_name: str,
    batch_size: int = 64,
    threshold: float | None = None,
    device: str | None = None,
) -> tuple[EvaluationArtifacts, pd.DataFrame]:
    checkpoint = load_inference_checkpoint(
        checkpoint_path,
        device=device,
        threshold=threshold,
    )
    manifest = load_feature_manifest_for_inference(feature_manifest_path)
    if "split" not in manifest.columns:
        raise InferenceError("Feature manifest must contain a split column for labeled evaluation.")
    if "label" not in manifest.columns:
        raise InferenceError("Feature manifest must contain a label column for labeled evaluation.")

    missing_feature_columns = [name for name in checkpoint.feature_names if name not in manifest.columns]
    is_hybrid_model = str(checkpoint.model_config.get("model_name")) == "cnn_lstm_audio"
    if missing_feature_columns and not is_hybrid_model:
        manifest = prepare_training_manifest(manifest)

    split_dataframe = manifest.loc[manifest["split"] == split_name].copy().reset_index(drop=True)
    if split_dataframe.empty:
        raise InferenceError(f"Feature manifest split '{split_name}' is empty.")

    if is_hybrid_model:
        dataset = VideoAudioSplitManifestDataset(split_dataframe)
        normalization_applied = False
        normalization_source = "not_applied"
    else:
        prepared_features, normalization_applied, normalization_source = prepare_feature_frame_for_inference(
            split_dataframe,
            feature_names=checkpoint.feature_names,
            normalization_stats=checkpoint.normalization_stats,
        )
        dataset = _FeatureOnlyDataset(
            torch.tensor(prepared_features.to_numpy(dtype="float32"), dtype=torch.float32)
        )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    collected_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            if "frames" in batch and "audio_features" in batch:
                logits = checkpoint.model(
                    batch["frames"].to(checkpoint.device),
                    batch["audio_features"].to(checkpoint.device),
                )
            else:
                features = batch["features"].to(checkpoint.device)
                logits = checkpoint.model(features)
            collected_logits.append(logits.detach().cpu())

    if not collected_logits:
        raise InferenceError(f"Split '{split_name}' produced zero rows to evaluate.")

    logits = torch.cat(collected_logits)
    labels = torch.tensor(
        pd.to_numeric(split_dataframe["label"], errors="raise").to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    metrics = compute_binary_classification_metrics(
        logits,
        labels,
        loss=float("nan"),
        threshold=checkpoint.threshold,
    )
    predictions = build_prediction_dataframe(
        split_dataframe,
        torch.sigmoid(logits),
        threshold=checkpoint.threshold,
    )

    return (
        EvaluationArtifacts(
            split_name=split_name,
            metrics=metrics,
            threshold=float(checkpoint.threshold),
            checkpoint_path=checkpoint.checkpoint_path,
            manifest_path=str(feature_manifest_path.expanduser().resolve()),
            normalization_applied=normalization_applied,
            normalization_source=normalization_source,
        ),
        predictions,
    )
