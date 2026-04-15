"""Validation helpers for labeled VOD clip datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = ("vod_id", "segment_id", "clip_path", "label")


@dataclass(frozen=True)
class ValidationIssue:
    """Represents one row-level dataset problem."""

    source_name: str
    row_number: int
    column: str
    message: str
    value: str

    def to_display_string(self) -> str:
        return (
            f"{self.source_name} row {self.row_number} [{self.column}]: "
            f"{self.message} (value={self.value!r})"
        )


def validate_required_columns(dataframe: pd.DataFrame, source_name: str) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        formatted = ", ".join(missing_columns)
        raise ValueError(f"{source_name} is missing required column(s): {formatted}")


def strip_string_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    for column in cleaned.columns:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return cleaned


def find_empty_required_values(
    dataframe: pd.DataFrame,
    source_name: str,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for row_index, row in dataframe.iterrows():
        row_number = int(row_index) + 2
        for column in required_columns:
            value = row[column]
            if pd.isna(value) or str(value).strip() == "":
                issues.append(
                    ValidationIssue(
                        source_name=source_name,
                        row_number=row_number,
                        column=column,
                        message="required value is missing",
                        value="" if pd.isna(value) else str(value),
                    )
                )
    return issues


def validate_binary_labels(dataframe: pd.DataFrame, source_name: str) -> tuple[pd.Series, list[ValidationIssue]]:
    normalized_labels: list[int | None] = []
    issues: list[ValidationIssue] = []

    for row_index, raw_value in dataframe["label"].items():
        row_number = int(row_index) + 2
        text_value = "" if pd.isna(raw_value) else str(raw_value).strip()

        if text_value not in {"0", "1"}:
            normalized_labels.append(None)
            issues.append(
                ValidationIssue(
                    source_name=source_name,
                    row_number=row_number,
                    column="label",
                    message="label must be binary 0 or 1",
                    value=text_value,
                )
            )
            continue

        normalized_labels.append(int(text_value))

    return pd.Series(normalized_labels, index=dataframe.index, dtype="Int64"), issues


def resolve_clip_path(
    raw_clip_path: str,
    csv_path: Path,
    clip_root: Path | None,
) -> tuple[Path | None, Path | None, str | None]:
    clip_path = Path(raw_clip_path)
    if clip_path.is_absolute():
        return None, None, "clip_path must be relative, not absolute"

    candidate_pairs: list[tuple[Path, Path]] = []
    if clip_root is not None:
        candidate_pairs.append((clip_root / clip_path, clip_root))
    candidate_pairs.append((csv_path.parent / clip_path, csv_path.parent))
    if clip_root is not None:
        candidate_pairs.append((clip_root / clip_path.name, clip_root))

    seen: set[Path] = set()
    for candidate, base_dir in candidate_pairs:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            try:
                relative_path = candidate.relative_to(base_dir)
            except ValueError:
                relative_path = clip_path
            return candidate.resolve(), relative_path, None

    return None, None, "referenced clip file does not exist"


def validate_clip_paths(
    dataframe: pd.DataFrame,
    source_name: str,
    csv_path: Path,
    clip_root: Path | None,
) -> tuple[pd.Series, pd.Series, list[ValidationIssue]]:
    resolved_paths: list[str | None] = []
    normalized_paths: list[str | None] = []
    issues: list[ValidationIssue] = []

    for row_index, raw_value in dataframe["clip_path"].items():
        row_number = int(row_index) + 2
        text_value = "" if pd.isna(raw_value) else str(raw_value).strip()
        resolved_path, normalized_path, error_message = resolve_clip_path(
            raw_clip_path=text_value,
            csv_path=csv_path,
            clip_root=clip_root,
        )

        if error_message is not None:
            resolved_paths.append(None)
            normalized_paths.append(None)
            issues.append(
                ValidationIssue(
                    source_name=source_name,
                    row_number=row_number,
                    column="clip_path",
                    message=error_message,
                    value=text_value,
                )
            )
            continue

        resolved_paths.append(str(resolved_path))
        normalized_paths.append(normalized_path.as_posix())

    return (
        pd.Series(resolved_paths, index=dataframe.index, dtype="string"),
        pd.Series(normalized_paths, index=dataframe.index, dtype="string"),
        issues,
    )
