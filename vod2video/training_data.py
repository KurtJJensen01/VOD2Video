"""Manifest-backed datasets and dataloaders for Phase 2B training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .training_config import DataConfig

DEFAULT_FEATURE_NAMES = (
    "start_time_seconds",
    "duration_seconds",
    "segment_index",
    "vod_index",
)


class TrainingDataError(ValueError):
    """Raised when a split manifest cannot be converted into training inputs."""


@dataclass(frozen=True)
class DatasetBundle:
    """Per-split datasets and dataloaders plus shared feature metadata."""

    dataframes: dict[str, pd.DataFrame]
    datasets: dict[str, Dataset]
    dataloaders: dict[str, DataLoader]
    feature_names: tuple[str, ...]
    input_dim: int
    normalization_stats: dict[str, dict[str, float]]


class SplitManifestDataset(Dataset):
    """Simple tensor dataset built from split manifest metadata columns.

    This placeholder dataset exists so Branch 2B can validate the training
    stack before the real feature extraction pipeline lands. It exposes the
    future model input contract as:

    - `features`: float tensor of shape `[feature_dim]`
    - `label`: float tensor scalar for BCE-style binary classification
    - `unique_id`: row identifier for logging/debugging
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        label_column: str = "label",
        feature_names: Iterable[str] = DEFAULT_FEATURE_NAMES,
    ) -> None:
        if label_column not in dataframe.columns:
            raise TrainingDataError(f"Manifest is missing label column: {label_column}")

        feature_name_list = tuple(feature_names)
        missing_columns = [name for name in feature_name_list if name not in dataframe.columns]
        if missing_columns:
            formatted = ", ".join(missing_columns)
            raise TrainingDataError(f"Manifest is missing derived feature column(s): {formatted}")

        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.label_column = label_column
        self.feature_names = feature_name_list

        feature_frame = self.dataframe.loc[:, self.feature_names].apply(pd.to_numeric, errors="coerce")
        if feature_frame.isna().any().any():
            bad_columns = feature_frame.columns[feature_frame.isna().any()].tolist()
            raise TrainingDataError(
                f"Feature columns contain missing or non-numeric values: {', '.join(bad_columns)}"
            )

        self.features = torch.tensor(feature_frame.to_numpy(dtype="float32"), dtype=torch.float32)
        self.labels = torch.tensor(
            pd.to_numeric(self.dataframe[self.label_column], errors="raise").to_numpy(dtype="float32"),
            dtype=torch.float32,
        )
        if "unique_id" in self.dataframe.columns:
            self.unique_ids = self.dataframe["unique_id"].astype("string").fillna("").tolist()
        else:
            self.unique_ids = [str(index) for index in range(len(self.dataframe))]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "features": self.features[index],
            "label": self.labels[index],
            "unique_id": self.unique_ids[index],
        }


def load_split_manifest(path: Path) -> pd.DataFrame:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def prepare_training_manifest(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Derive lightweight numeric features from split metadata.

    These fields are intentionally simple placeholders so the training
    framework can run before full frame/feature extraction exists.
    """

    work = dataframe.copy()
    if "label" not in work.columns:
        raise TrainingDataError("Manifest must contain a label column.")
    if "split" not in work.columns:
        raise TrainingDataError("Manifest must contain a split column.")

    if "start_time_seconds" in work.columns:
        work["start_time_seconds"] = pd.to_numeric(work["start_time_seconds"], errors="coerce")
    else:
        work["start_time_seconds"] = pd.to_numeric(
            work["segment_id"].astype("string").str.extract(r"(\d+)$", expand=False),
            errors="coerce",
        )

    if "end_time_seconds" in work.columns:
        work["end_time_seconds"] = pd.to_numeric(work["end_time_seconds"], errors="coerce")
    else:
        work["end_time_seconds"] = work["start_time_seconds"] + 1.0

    work["duration_seconds"] = work["end_time_seconds"] - work["start_time_seconds"]
    work["segment_index"] = pd.to_numeric(
        work["segment_id"].astype("string").str.extract(r"(\d+)$", expand=False),
        errors="coerce",
    )

    vod_values = work["vod_id"].astype("string").tolist()
    unique_vods = {vod_id: index for index, vod_id in enumerate(dict.fromkeys(vod_values))}
    work["vod_index"] = [float(unique_vods[vod_id]) for vod_id in vod_values]

    required_columns = ("start_time_seconds", "duration_seconds", "segment_index", "vod_index")
    for column in required_columns:
        if work[column].isna().any():
            bad_rows = work.loc[work[column].isna(), "unique_id"].astype("string").head(5).tolist()
            raise TrainingDataError(
                f"Could not derive {column} for row(s): {', '.join(bad_rows)}"
            )

    work["label"] = pd.to_numeric(work["label"], errors="raise").astype(int)
    return work


def build_dataloaders_from_manifest(
    data_config: DataConfig,
    *,
    feature_names: Iterable[str] = DEFAULT_FEATURE_NAMES,
) -> DatasetBundle:
    manifest = load_split_manifest(data_config.split_manifest_path)
    prepared_manifest = prepare_training_manifest(manifest)

    if data_config.split_column not in prepared_manifest.columns:
        raise TrainingDataError(
            f"Manifest is missing split column: {data_config.split_column}"
        )

    split_names = (
        data_config.train_split_name,
        data_config.val_split_name,
        data_config.test_split_name,
    )

    dataframes: dict[str, pd.DataFrame] = {}
    datasets: dict[str, Dataset] = {}
    dataloaders: dict[str, DataLoader] = {}
    feature_name_list = tuple(feature_names)
    train_split_df = prepared_manifest.loc[
        prepared_manifest[data_config.split_column] == data_config.train_split_name
    ].copy()
    if train_split_df.empty:
        raise TrainingDataError(f"Manifest split '{data_config.train_split_name}' is empty.")

    train_feature_frame = train_split_df.loc[:, feature_name_list].apply(pd.to_numeric, errors="coerce")
    feature_means = train_feature_frame.mean()
    feature_stds = train_feature_frame.std(ddof=0).replace(0.0, 1.0).fillna(1.0)

    for split_name in split_names:
        split_df = prepared_manifest.loc[
            prepared_manifest[data_config.split_column] == split_name
        ].copy()
        if split_df.empty:
            raise TrainingDataError(f"Manifest split '{split_name}' is empty.")

        feature_frame = split_df.loc[:, feature_name_list].apply(pd.to_numeric, errors="coerce")
        normalized_features = (feature_frame - feature_means) / feature_stds
        split_df = split_df.astype({column: "float64" for column in feature_name_list})
        for column in feature_name_list:
            split_df[column] = normalized_features[column].astype("float32")

        dataset = SplitManifestDataset(
            split_df,
            label_column=data_config.label_column,
            feature_names=feature_name_list,
        )
        dataframes[split_name] = split_df
        datasets[split_name] = dataset
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=data_config.shuffle_train if split_name == data_config.train_split_name else False,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
        )

    return DatasetBundle(
        dataframes=dataframes,
        datasets=datasets,
        dataloaders=dataloaders,
        feature_names=feature_name_list,
        input_dim=len(feature_name_list),
        normalization_stats={
            "means": {column: float(value) for column, value in feature_means.items()},
            "stds": {column: float(value) for column, value in feature_stds.items()},
        },
    )
