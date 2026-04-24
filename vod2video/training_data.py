"""Manifest-backed datasets and dataloaders for Phase 2B training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .clip_features import (
    AudioExtractionConfig,
    AudioToolStatus,
    extract_audio_features,
    resolve_audio_tool_status,
)
from .training_config import DataConfig

DEFAULT_FEATURE_NAMES = (
    "start_time_seconds",
    "duration_seconds",
    "segment_index",
    "vod_index",
)

HYBRID_AUDIO_FEATURE_NAMES = (
    "rms_mean",
    "rms_std",
    "peak_amplitude",
    "silence_ratio",
    "energy_mean",
    "energy_std",
    "energy_max",
)

HYBRID_AUDIO_FEATURE_COLUMNS = (
    "audio_rms_mean",
    "audio_rms_std",
    "audio_peak_amplitude",
    "audio_silence_ratio",
    "audio_energy_mean",
    "audio_energy_std",
    "audio_energy_max",
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
    """Simple tensor dataset built from numeric manifest columns.

    It supports both the original placeholder manifest-derived columns and
    later precomputed real clip feature columns. It exposes the shared model
    input contract as:

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


class VideoAudioSplitManifestDataset(Dataset):
    """Dataset that decodes clip frames and ffmpeg-derived audio features on demand."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        label_column: str = "label",
        frame_count: int = 10,
        frame_size: int = 224,
        audio_config: AudioExtractionConfig | None = None,
        audio_tool_status: AudioToolStatus | None = None,
    ) -> None:
        if label_column not in dataframe.columns:
            raise TrainingDataError(f"Manifest is missing label column: {label_column}")
        if "resolved_clip_path" not in dataframe.columns:
            raise TrainingDataError("Manifest is missing required column: resolved_clip_path")

        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.label_column = label_column
        self.frame_count = int(frame_count)
        self.frame_size = int(frame_size)
        self.audio_config = audio_config or AudioExtractionConfig()
        self.audio_tool_status = audio_tool_status or resolve_audio_tool_status(self.audio_config)

        self.labels = torch.tensor(
            pd.to_numeric(self.dataframe[self.label_column], errors="raise").to_numpy(dtype="float32"),
            dtype=torch.float32,
        )
        self.clip_paths = [
            str(Path(str(value)).expanduser().resolve())
            for value in self.dataframe["resolved_clip_path"].tolist()
        ]
        missing_paths = [path for path in self.clip_paths if not Path(path).exists()]
        if missing_paths:
            preview = ", ".join(missing_paths[:5])
            raise TrainingDataError(f"Resolved clip path(s) not found: {preview}")

        if "unique_id" in self.dataframe.columns:
            self.unique_ids = self.dataframe["unique_id"].astype("string").fillna("").tolist()
        else:
            self.unique_ids = [str(index) for index in range(len(self.dataframe))]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        clip_path = Path(self.clip_paths[index])
        return {
            "frames": _read_evenly_spaced_rgb_frames(
                clip_path,
                frame_count=self.frame_count,
                frame_size=self.frame_size,
            ),
            "audio_features": _extract_hybrid_audio_tensor(
                clip_path,
                audio_config=self.audio_config,
                audio_tool_status=self.audio_tool_status,
            ),
            "label": self.labels[index],
            "clip_path": str(clip_path),
            "unique_id": self.unique_ids[index],
        }


def _read_evenly_spaced_rgb_frames(
    clip_path: Path,
    *,
    frame_count: int,
    frame_size: int,
) -> torch.Tensor:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise TrainingDataError(f"Could not open clip: {clip_path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise TrainingDataError(f"Clip has no readable frames: {clip_path}")

        indices = np.linspace(0, total_frames - 1, num=frame_count, dtype=np.int32)
        frames: list[np.ndarray] = []
        last_frame: np.ndarray | None = None
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                if last_frame is None:
                    continue
                frame_rgb = last_frame.copy()
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                last_frame = frame_rgb

            resized = cv2.resize(frame_rgb, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
            frames.append(resized.astype(np.float32) / 255.0)

        if not frames:
            raise TrainingDataError(f"Failed to decode sampled frames from: {clip_path}")

        while len(frames) < frame_count:
            frames.append(frames[-1].copy())

        frame_array = np.stack(frames[:frame_count], axis=0)
        frame_tensor = torch.from_numpy(frame_array).permute(0, 3, 1, 2).contiguous()
        return frame_tensor.to(dtype=torch.float32)
    finally:
        capture.release()


def _extract_hybrid_audio_tensor(
    clip_path: Path,
    *,
    audio_config: AudioExtractionConfig,
    audio_tool_status: AudioToolStatus,
) -> torch.Tensor:
    try:
        features = extract_audio_features(
            clip_path,
            audio_config=audio_config,
            tool_status=audio_tool_status,
        )
    except Exception:
        return torch.zeros(len(HYBRID_AUDIO_FEATURE_COLUMNS), dtype=torch.float32)

    if float(features.get("audio_available", 0.0)) <= 0.0:
        return torch.zeros(len(HYBRID_AUDIO_FEATURE_COLUMNS), dtype=torch.float32)

    values = [float(features.get(column, 0.0)) for column in HYBRID_AUDIO_FEATURE_COLUMNS]
    return torch.tensor(values, dtype=torch.float32)


def load_split_manifest(path: Path) -> pd.DataFrame:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def prepare_training_manifest(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Derive lightweight numeric placeholder features from split metadata.

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


def build_video_audio_dataloaders_from_manifest(
    data_config: DataConfig,
    *,
    audio_config: AudioExtractionConfig | None = None,
) -> DatasetBundle:
    manifest = load_split_manifest(data_config.split_manifest_path)
    if data_config.label_column not in manifest.columns:
        raise TrainingDataError(f"Manifest is missing label column: {data_config.label_column}")
    if data_config.split_column not in manifest.columns:
        raise TrainingDataError(f"Manifest is missing split column: {data_config.split_column}")
    if "resolved_clip_path" not in manifest.columns:
        raise TrainingDataError("Manifest is missing required column: resolved_clip_path")

    split_names = (
        data_config.train_split_name,
        data_config.val_split_name,
        data_config.test_split_name,
    )
    active_audio_config = audio_config or AudioExtractionConfig()
    audio_tool_status = resolve_audio_tool_status(active_audio_config)

    dataframes: dict[str, pd.DataFrame] = {}
    datasets: dict[str, Dataset] = {}
    dataloaders: dict[str, DataLoader] = {}

    for split_name in split_names:
        split_df = manifest.loc[manifest[data_config.split_column] == split_name].copy()
        if split_df.empty:
            raise TrainingDataError(f"Manifest split '{split_name}' is empty.")

        dataset = VideoAudioSplitManifestDataset(
            split_df,
            label_column=data_config.label_column,
            frame_count=data_config.frame_count,
            frame_size=data_config.frame_size,
            audio_config=active_audio_config,
            audio_tool_status=audio_tool_status,
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
        feature_names=HYBRID_AUDIO_FEATURE_NAMES,
        input_dim=len(HYBRID_AUDIO_FEATURE_NAMES),
        normalization_stats={},
    )


def compute_positive_class_weight_from_dataframe(
    dataframe: pd.DataFrame,
    *,
    split_column: str = "split",
    label_column: str = "label",
    train_split_name: str = "train",
    cap: float = 10.0,
) -> float:
    if split_column not in dataframe.columns or label_column not in dataframe.columns:
        raise TrainingDataError(f"Manifest must contain {split_column} and {label_column} columns.")
    train_rows = dataframe.loc[dataframe[split_column] == train_split_name].copy()
    if train_rows.empty:
        raise TrainingDataError(f"Manifest split '{train_split_name}' is empty.")

    labels = pd.to_numeric(train_rows[label_column], errors="raise").astype(int)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if positive_count == 0:
        raise TrainingDataError("Train split contains no positive examples; cannot derive class weight.")
    return min(float(negative_count / positive_count), float(cap))


def compute_positive_class_weight_from_manifest(
    manifest_path: Path,
    *,
    split_column: str = "split",
    label_column: str = "label",
    train_split_name: str = "train",
    cap: float = 10.0,
) -> float:
    return compute_positive_class_weight_from_dataframe(
        load_split_manifest(manifest_path),
        split_column=split_column,
        label_column=label_column,
        train_split_name=train_split_name,
        cap=cap,
    )
