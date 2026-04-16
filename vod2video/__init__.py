"""Reusable helpers for the VOD2Video project."""

from .dataset_loader import (
    DatasetSummary,
    DatasetValidationError,
    LabeledDatasetSource,
    LoadedDataset,
    format_summary,
    load_labeled_dataset,
)
from .dataset_split import (
    DatasetSplit,
    DatasetSplitError,
    SplitConfig,
    SplitSummary,
    format_split_summaries,
    split_labeled_dataset,
    write_split_manifests,
)
from .metrics import BinaryClassificationMetrics, compute_binary_classification_metrics
from .models import MLPBaselineModel, build_model
from .training import train_model, train_one_epoch, validate_model
from .training_config import CheckpointConfig, DataConfig, ModelConfig, TrainingConfig
from .training_data import (
    DEFAULT_FEATURE_NAMES,
    DatasetBundle,
    SplitManifestDataset,
    TrainingDataError,
    build_dataloaders_from_manifest,
    load_split_manifest,
    prepare_training_manifest,
)

__all__ = [
    "BinaryClassificationMetrics",
    "CheckpointConfig",
    "DEFAULT_FEATURE_NAMES",
    "DataConfig",
    "DatasetBundle",
    "DatasetSummary",
    "DatasetValidationError",
    "DatasetSplit",
    "DatasetSplitError",
    "LabeledDatasetSource",
    "LoadedDataset",
    "MLPBaselineModel",
    "ModelConfig",
    "SplitConfig",
    "SplitManifestDataset",
    "SplitSummary",
    "TrainingConfig",
    "TrainingDataError",
    "build_dataloaders_from_manifest",
    "build_model",
    "compute_binary_classification_metrics",
    "format_summary",
    "format_split_summaries",
    "load_labeled_dataset",
    "load_split_manifest",
    "prepare_training_manifest",
    "split_labeled_dataset",
    "train_model",
    "train_one_epoch",
    "validate_model",
    "write_split_manifests",
]
