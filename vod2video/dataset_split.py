"""Leakage-resistant train/validation/test splitting for labeled VOD clips.

This module implements a block-based, time-aware split strategy for sparse clip
labels sampled from longer VOD timelines. Rather than randomly splitting rows,
it first groups nearby clips within each VOD into contiguous time blocks using a
configurable neighbor window, then assigns whole blocks to `train`, `val`, or
`test`. Keeping neighboring clips in the same partition reduces obvious leakage
from near-identical moments appearing across multiple splits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Mapping

import pandas as pd

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_REQUIRED_COLUMNS = ("vod_id", "label", "unique_id")


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for repeatable dataset splitting.

    `neighbor_window_seconds` controls the block-building step and stays fully
    configurable so later experiments can tighten or loosen how aggressively
    nearby clips are kept together.
    """

    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    neighbor_window_seconds: int = 60
    split_column: str = "split"
    block_column: str = "split_block_id"

    def fractions_by_split(self) -> dict[str, float]:
        return {
            "train": self.train_fraction,
            "val": self.val_fraction,
            "test": self.test_fraction,
        }


@dataclass(frozen=True)
class SplitSummary:
    total_rows: int
    class_balance: dict[int, int]
    rows_per_vod: dict[str, int]
    block_count: int


@dataclass(frozen=True)
class DatasetSplit:
    """Full split assignment plus per-split views and summaries.

    The combined dataframe preserves the Phase 1A loader columns and adds:
    - `split`: the assigned dataset partition
    - `split_block_id`: the contiguous time block used to avoid neighbor leakage
    """

    dataframe: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    summaries: dict[str, SplitSummary]
    config: SplitConfig


class DatasetSplitError(ValueError):
    """Raised when split inputs or configuration are invalid."""


def _validate_config(config: SplitConfig) -> None:
    fractions = config.fractions_by_split()
    if any(value <= 0 for value in fractions.values()):
        raise DatasetSplitError("All split fractions must be greater than zero.")

    total_fraction = sum(fractions.values())
    if abs(total_fraction - 1.0) > 1e-9:
        raise DatasetSplitError(
            f"Split fractions must sum to 1.0, got {total_fraction:.6f}."
        )

    if config.neighbor_window_seconds < 0:
        raise DatasetSplitError("neighbor_window_seconds must be zero or greater.")


def _coerce_time_series(dataframe: pd.DataFrame) -> pd.Series:
    if "start_time_seconds" in dataframe.columns:
        series = pd.to_numeric(dataframe["start_time_seconds"], errors="coerce")
        if series.isna().any():
            bad_rows = dataframe.loc[series.isna(), "unique_id"].head(5).tolist()
            raise DatasetSplitError(
                "start_time_seconds contains missing or non-numeric values for "
                f"{bad_rows}."
            )
        return series.astype(float)

    if "segment_id" not in dataframe.columns:
        raise DatasetSplitError(
            "Dataset must include either start_time_seconds or segment_id for time-aware splitting."
        )

    extracted = dataframe["segment_id"].astype("string").str.extract(r"(\d+)$", expand=False)
    series = pd.to_numeric(extracted, errors="coerce")
    if series.isna().any():
        bad_rows = dataframe.loc[series.isna(), "unique_id"].head(5).tolist()
        raise DatasetSplitError(
            "Could not derive an ordering value from segment_id for "
            f"{bad_rows}."
        )
    return series.astype(float)


def _validate_required_columns(dataframe: pd.DataFrame) -> None:
    missing = [column for column in DEFAULT_REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        formatted = ", ".join(missing)
        raise DatasetSplitError(f"Dataset is missing required split column(s): {formatted}")


def _build_block_table(
    dataframe: pd.DataFrame,
    config: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = dataframe.copy()
    _validate_required_columns(work)
    work["_split_time_value"] = _coerce_time_series(work)
    work = work.sort_values(["vod_id", "_split_time_value", "unique_id"], kind="mergesort").reset_index()

    block_labels: list[str] = []
    for vod_id, group in work.groupby("vod_id", sort=False):
        block_index = -1
        previous_time: float | None = None

        for current_time in group["_split_time_value"].tolist():
            current_time = float(current_time)
            if previous_time is None or (current_time - previous_time) > config.neighbor_window_seconds:
                block_index += 1
            block_labels.append(f"{vod_id}_block_{block_index:04d}")
            previous_time = current_time

    work[config.block_column] = block_labels

    block_table = (
        work.groupby(["vod_id", config.block_column], sort=False)
        .agg(
            row_count=("unique_id", "size"),
            positive_count=("label", "sum"),
            min_time=("_split_time_value", "min"),
            max_time=("_split_time_value", "max"),
        )
        .reset_index()
    )
    block_table["negative_count"] = block_table["row_count"] - block_table["positive_count"]
    return work, block_table


def _assign_blocks_within_vod(vod_blocks: pd.DataFrame, config: SplitConfig) -> dict[str, str]:
    fractions = config.fractions_by_split()
    total_rows = int(vod_blocks["row_count"].sum())
    total_positives = int(vod_blocks["positive_count"].sum())

    target_rows = {
        split_name: total_rows * fraction for split_name, fraction in fractions.items()
    }
    target_positives = {
        split_name: total_positives * fraction for split_name, fraction in fractions.items()
    }

    shuffled_blocks = list(vod_blocks.to_dict("records"))
    vod_seed = sum(ord(character) for character in str(vod_blocks["vod_id"].iloc[0]))
    rng = random.Random(config.seed + vod_seed)
    rng.shuffle(shuffled_blocks)
    ordered_blocks = sorted(
        shuffled_blocks,
        key=lambda row: (
            int(row["positive_count"]) > 0,
            int(row["row_count"]),
            float(row["min_time"]),
        ),
        reverse=True,
    )

    split_rows = {split_name: 0 for split_name in SPLIT_NAMES}
    split_positives = {split_name: 0 for split_name in SPLIT_NAMES}
    split_blocks = {split_name: 0 for split_name in SPLIT_NAMES}
    assignments: dict[str, str] = {}

    for index, block_row in enumerate(ordered_blocks):
        remaining_including_current = len(ordered_blocks) - index
        empty_splits = [name for name in SPLIT_NAMES if split_blocks[name] == 0]
        candidate_splits = list(SPLIT_NAMES)
        if empty_splits and remaining_including_current <= len(empty_splits):
            candidate_splits = empty_splits

        block_rows = int(block_row["row_count"])
        block_positives = int(block_row["positive_count"])

        def candidate_key(split_name: str) -> tuple[float, float, int, int]:
            new_row_fill = (split_rows[split_name] + block_rows) / max(target_rows[split_name], 1.0)
            new_positive_fill = 0.0
            if total_positives > 0:
                new_positive_fill = (
                    (split_positives[split_name] + block_positives)
                    / max(target_positives[split_name], 1.0)
                )

            if block_positives > 0 and total_positives > 0:
                return (
                    max(new_row_fill, new_positive_fill),
                    new_positive_fill,
                    split_blocks[split_name],
                    SPLIT_NAMES.index(split_name),
                )

            return (
                new_row_fill,
                new_positive_fill,
                split_blocks[split_name],
                SPLIT_NAMES.index(split_name),
            )

        best_split = min(candidate_splits, key=candidate_key)

        block_id = str(block_row[config.block_column])
        assignments[block_id] = best_split
        split_rows[best_split] += block_rows
        split_positives[best_split] += block_positives
        split_blocks[best_split] += 1

    return assignments


def build_split_summary(
    dataframe: pd.DataFrame,
    *,
    block_column: str,
) -> SplitSummary:
    class_balance = {
        int(label): int(count)
        for label, count in dataframe["label"].value_counts(sort=False).items()
    }
    rows_per_vod = {
        str(vod_id): int(count)
        for vod_id, count in dataframe["vod_id"].value_counts(sort=False).items()
    }
    return SplitSummary(
        total_rows=int(len(dataframe)),
        class_balance=class_balance,
        rows_per_vod=rows_per_vod,
        block_count=int(dataframe[block_column].nunique()) if len(dataframe) else 0,
    )


def format_split_summaries(summaries: Mapping[str, SplitSummary]) -> str:
    lines = ["Split summary"]
    for split_name in SPLIT_NAMES:
        summary = summaries[split_name]
        class_text = ", ".join(
            f"{label}={count}" for label, count in summary.class_balance.items()
        ) or "none"
        vod_text = ", ".join(
            f"{vod_id}={count}" for vod_id, count in summary.rows_per_vod.items()
        ) or "none"
        lines.extend(
            [
                f"  {split_name}",
                f"    total rows: {summary.total_rows}",
                f"    class balance: {class_text}",
                f"    rows per vod_id: {vod_text}",
                f"    block count: {summary.block_count}",
            ]
        )
    return "\n".join(lines)


def split_labeled_dataset(
    dataframe: pd.DataFrame,
    *,
    config: SplitConfig | None = None,
) -> DatasetSplit:
    """Split labeled clip rows into leakage-resistant train/val/test partitions.

    Expected behavior:
    - Preserve the incoming dataset rows and columns from Phase 1A.
    - Build contiguous time-aware blocks within each `vod_id`.
    - Assign whole blocks, not individual rows, to a single split.
    - Produce deterministic assignments when the same input and seed are used.
    """

    active_config = config or SplitConfig()
    _validate_config(active_config)

    work, block_table = _build_block_table(dataframe, active_config)

    assignments: dict[str, str] = {}
    for _, vod_blocks in block_table.groupby("vod_id", sort=False):
        assignments.update(_assign_blocks_within_vod(vod_blocks, active_config))

    work[active_config.split_column] = work[active_config.block_column].map(assignments)
    if work[active_config.split_column].isna().any():
        missing_ids = work.loc[work[active_config.split_column].isna(), active_config.block_column].unique()
        raise DatasetSplitError(
            f"Failed to assign split labels for block(s): {', '.join(map(str, missing_ids[:5]))}"
        )

    output = (
        work.drop(columns=["_split_time_value"])
        .sort_values(["vod_id", active_config.block_column, "unique_id"], kind="mergesort")
        .set_index("index")
        .sort_index()
    )
    output.index.name = None

    splits = {
        split_name: output.loc[output[active_config.split_column] == split_name].copy()
        for split_name in SPLIT_NAMES
    }
    summaries = {
        split_name: build_split_summary(split_df, block_column=active_config.block_column)
        for split_name, split_df in splits.items()
    }
    return DatasetSplit(
        dataframe=output.copy(),
        splits=splits,
        summaries=summaries,
        config=active_config,
    )


def write_split_manifests(dataset_split: DatasetSplit, output_dir: Path) -> dict[str, Path]:
    """Write optional combined and per-split CSV manifests for later training code.

    These files are generated outputs for convenience, not required source
    inputs for the split logic itself.
    """

    output_path = output_dir.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_paths = {
        "all": output_path / "all_splits.csv",
        "train": output_path / "train.csv",
        "val": output_path / "val.csv",
        "test": output_path / "test.csv",
        "config": output_path / "split_config.json",
        "summary": output_path / "split_summary.json",
    }

    dataset_split.dataframe.to_csv(manifest_paths["all"], index=False)
    for split_name in SPLIT_NAMES:
        dataset_split.splits[split_name].to_csv(manifest_paths[split_name], index=False)

    manifest_paths["config"].write_text(
        json.dumps(asdict(dataset_split.config), indent=2),
        encoding="utf-8",
    )
    manifest_paths["summary"].write_text(
        json.dumps(
            {
                split_name: asdict(summary)
                for split_name, summary in dataset_split.summaries.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_paths
