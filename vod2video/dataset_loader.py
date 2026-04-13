"""Dataset loader for Phase 1A labeled VOD clip data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import pandas as pd

from .validation import (
    ValidationIssue,
    find_empty_required_values,
    strip_string_columns,
    validate_binary_labels,
    validate_clip_paths,
    validate_required_columns,
)

OUTPUT_COLUMNS = (
    "vod_id",
    "segment_id",
    "unique_id",
    "label",
    "clip_path",
    "resolved_clip_path",
    "source_csv",
)


@dataclass(frozen=True)
class LabeledDatasetSource:
    """One labeled CSV plus the root used to resolve its clip paths."""

    csv_path: Path
    clip_root: Path | None = None
    source_name: str | None = None

    def resolved_csv_path(self) -> Path:
        return self.csv_path.expanduser().resolve()

    def resolved_clip_root(self) -> Path | None:
        if self.clip_root is None:
            return None
        return self.clip_root.expanduser().resolve()

    def display_name(self) -> str:
        return self.source_name or self.csv_path.name


@dataclass(frozen=True)
class DatasetSummary:
    total_rows: int
    rows_per_vod: dict[str, int]
    class_balance: dict[int, int]
    missing_or_bad_paths: int
    source_row_counts: dict[str, int]
    unique_vod_count: int


@dataclass(frozen=True)
class LoadedDataset:
    """Validated combined dataset plus summary metadata."""

    dataframe: pd.DataFrame
    summary: DatasetSummary
    sources: tuple[LabeledDatasetSource, ...]


class DatasetValidationError(ValueError):
    """Raised when a labeled dataset fails validation."""


def _validate_source_paths(source: LabeledDatasetSource) -> tuple[Path, Path | None]:
    csv_path = source.resolved_csv_path()
    clip_root = source.resolved_clip_root()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if clip_root is not None and not clip_root.exists():
        raise FileNotFoundError(f"Clip root not found for {source.display_name()}: {clip_root}")

    return csv_path, clip_root


def _format_issue_block(issues: list[ValidationIssue], max_issue_examples: int) -> str:
    header = f"Found {len(issues)} dataset validation issue(s)."
    examples = [issue.to_display_string() for issue in issues[:max_issue_examples]]
    lines = [header, *examples]
    if len(issues) > max_issue_examples:
        remaining = len(issues) - max_issue_examples
        lines.append(f"... and {remaining} more issue(s).")
    return "\n".join(lines)


def _warn_before_failing(issues: list[ValidationIssue], max_issue_examples: int) -> None:
    warnings.warn(
        _format_issue_block(issues, max_issue_examples=max_issue_examples),
        stacklevel=2,
    )


def _load_one_source(
    source: LabeledDatasetSource,
    max_issue_examples: int,
    warn_on_error: bool,
) -> pd.DataFrame:
    """Load and validate one labeled CSV source."""

    csv_path, clip_root = _validate_source_paths(source)
    dataframe = pd.read_csv(csv_path, dtype="string", keep_default_na=False)
    dataframe = strip_string_columns(dataframe)

    validate_required_columns(dataframe, source.display_name())

    issues = find_empty_required_values(dataframe, source.display_name())

    label_series, label_issues = validate_binary_labels(dataframe, source.display_name())
    resolved_paths, normalized_paths, path_issues = validate_clip_paths(
        dataframe=dataframe,
        source_name=source.display_name(),
        csv_path=csv_path,
        clip_root=clip_root,
    )

    issues.extend(label_issues)
    issues.extend(path_issues)

    if issues:
        if warn_on_error:
            _warn_before_failing(issues, max_issue_examples=max_issue_examples)
        raise DatasetValidationError(_format_issue_block(issues, max_issue_examples=max_issue_examples))

    dataframe = dataframe.copy()
    dataframe["vod_id"] = dataframe["vod_id"].astype("string")
    dataframe["segment_id"] = dataframe["segment_id"].astype("string")
    dataframe["label"] = label_series.astype(int)
    dataframe["clip_path"] = normalized_paths
    dataframe["resolved_clip_path"] = resolved_paths
    # `segment_id` is only unique within a VOD, so later code should key off `unique_id`.
    dataframe["unique_id"] = dataframe["vod_id"] + "_" + dataframe["segment_id"]
    dataframe["source_csv"] = str(csv_path)

    return dataframe


def _check_for_duplicate_unique_ids(
    dataframe: pd.DataFrame,
    max_issue_examples: int,
    warn_on_error: bool,
) -> None:
    duplicate_rows = dataframe[dataframe["unique_id"].duplicated(keep=False)]
    if duplicate_rows.empty:
        return

    issues = [
        ValidationIssue(
            source_name=Path(row["source_csv"]).name,
            row_number=int(index) + 2,
            column="unique_id",
            message="duplicate unique_id detected in combined dataset",
            value=row["unique_id"],
        )
        for index, row in duplicate_rows.iterrows()
    ]

    if warn_on_error:
        _warn_before_failing(issues, max_issue_examples=max_issue_examples)
    raise DatasetValidationError(_format_issue_block(issues, max_issue_examples=max_issue_examples))


def build_dataset_summary(dataframe: pd.DataFrame) -> DatasetSummary:
    rows_per_vod = {
        str(vod_id): int(count)
        for vod_id, count in dataframe["vod_id"].value_counts(sort=False).items()
    }
    class_balance = {
        int(label): int(count)
        for label, count in dataframe["label"].value_counts(sort=False).items()
    }
    source_row_counts = {
        Path(source_csv).name: int(count)
        for source_csv, count in dataframe["source_csv"].value_counts(sort=False).items()
    }

    return DatasetSummary(
        total_rows=int(len(dataframe)),
        rows_per_vod=rows_per_vod,
        class_balance=class_balance,
        missing_or_bad_paths=int(dataframe["resolved_clip_path"].isna().sum()),
        source_row_counts=source_row_counts,
        unique_vod_count=int(dataframe["vod_id"].nunique()),
    )


def format_summary(summary: DatasetSummary) -> str:
    rows_per_vod_text = ", ".join(
        f"{vod_id}={count}" for vod_id, count in summary.rows_per_vod.items()
    )
    class_balance_text = ", ".join(
        f"{label}={count}" for label, count in summary.class_balance.items()
    )
    source_text = ", ".join(
        f"{source_name}={count}" for source_name, count in summary.source_row_counts.items()
    )

    return "\n".join(
        [
            "Dataset summary",
            f"  total rows: {summary.total_rows}",
            f"  unique vods: {summary.unique_vod_count}",
            f"  rows per vod_id: {rows_per_vod_text}",
            f"  class balance: {class_balance_text}",
            f"  rows per source CSV: {source_text}",
            f"  missing/bad paths: {summary.missing_or_bad_paths}",
        ]
    )


def load_labeled_dataset(
    sources: Iterable[LabeledDatasetSource],
    *,
    max_issue_examples: int = 10,
    warn_on_error: bool = True,
) -> LoadedDataset:
    """Load, validate, and combine one or more labeled dataset sources.

    The returned dataframe is intended to be reusable by later branches and always
    includes the Phase 1A contract columns:
    `vod_id`, `segment_id`, `unique_id`, `label`, `clip_path`,
    `resolved_clip_path`, and `source_csv`.

    `clip_path` remains a normalized relative path, while `resolved_clip_path`
    is the absolute on-disk file path that passed validation.
    """

    source_list = tuple(sources)
    if not source_list:
        raise ValueError("At least one dataset source is required.")

    dataframes = [
        _load_one_source(
            source=source,
            max_issue_examples=max_issue_examples,
            warn_on_error=warn_on_error,
        )
        for source in source_list
    ]
    combined = pd.concat(dataframes, ignore_index=True)
    _check_for_duplicate_unique_ids(
        combined,
        max_issue_examples=max_issue_examples,
        warn_on_error=warn_on_error,
    )
    missing_output_columns = [column for column in OUTPUT_COLUMNS if column not in combined.columns]
    if missing_output_columns:
        formatted = ", ".join(missing_output_columns)
        raise DatasetValidationError(
            f"Combined dataset is missing required output column(s): {formatted}"
        )

    summary = build_dataset_summary(combined)
    return LoadedDataset(dataframe=combined, summary=summary, sources=source_list)
