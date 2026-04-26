"""Final demo package assembly for Branch 5B.

Reads the four Branch 4C demo-selection CSVs (true positives, false positives,
false negatives, borderline) and produces a single self-contained folder of
copied clips plus a unified manifest CSV and a summary JSON. This is the
physical hand-off artifact for Phase 5B.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import shutil
from pathlib import Path

import pandas as pd

# Keys are the final category folder names; values are the Branch 4C CSV files
# we read from. We intentionally skip top_ranked_highlights.csv because the
# Phase 5B spec only calls for TP / FP / FN / borderline.
DEFAULT_SOURCE_CSV_NAMES: dict[str, str] = {
    "true_positives": "top_true_positives.csv",
    "false_positives": "top_false_positives.csv",
    "false_negatives": "top_false_negatives.csv",
    "borderline": "borderline_examples.csv",
}

# Manifest columns are placed in this order when present. Any extra columns
# carried over from the Branch 4C CSVs are appended after these for traceability.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "final_category",
    "category_rank",
    "unique_id",
    "vod_id",
    "segment_id",
    "label",
    "predicted_probability",
    "predicted_class",
    "review_group",
    "source_clip_path",
    "packaged_clip_path",
)


class FinalDemoPackageError(ValueError):
    """Raised when the final demo package cannot be built."""


@dataclass(frozen=True)
class FinalDemoPackageSummary:
    branch: str
    source_dir: str
    output_dir: str
    threshold: float
    allow_missing: bool
    category_counts: dict[str, int]
    total_clips_copied: int
    missing_source_files: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_category_csv(source_dir: Path, file_name: str, category: str) -> pd.DataFrame:
    csv_path = source_dir / file_name
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Branch 4C CSV not found for category '{category}': {csv_path}"
        )
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return frame
    if "unique_id" not in frame.columns:
        raise FinalDemoPackageError(
            f"Branch 4C CSV {csv_path} is missing required unique_id column."
        )
    frame = frame.copy().reset_index(drop=True)
    frame["final_category"] = category
    # `demo_rank` is the per-group rank set by Branch 4C. Surface it under a
    # consistent name so downstream consumers don't need to know 4C internals.
    if "demo_rank" in frame.columns:
        frame["category_rank"] = frame["demo_rank"]
    else:
        frame["category_rank"] = range(1, len(frame) + 1)
    return frame


def _resolve_source_clip(
    row: pd.Series,
    repo_root: Path,
    clip_root: Path,
) -> Path | None:
    """Return the on-disk clip path for a row, or None if it cannot be located.

    Tries `resolved_clip_path` first (absolute path saved by Branch 1A/4C),
    then falls back to `clip_path` resolved against `clip_root` (the canonical
    labeling root, e.g. `labeling_test/`), and finally against the repo root.
    The fallbacks matter when the package is rebuilt on a different machine
    where the absolute paths from a previous run no longer exist.
    """

    candidates: list[Path] = []
    resolved = row.get("resolved_clip_path")
    if isinstance(resolved, str) and resolved:
        candidates.append(Path(resolved))
    relative = row.get("clip_path")
    if isinstance(relative, str) and relative:
        # Source CSVs may use either Windows or POSIX separators.
        normalized = relative.replace("\\", "/")
        candidates.append(clip_root / normalized)
        candidates.append(repo_root / normalized)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _copy_clip(source: Path, destination_dir: Path, unique_id: str) -> Path:
    # Prefix the original file name with unique_id so files from different
    # VODs that share a segment_id never collide inside one category folder.
    destination = destination_dir / f"{unique_id}__{source.name}"
    shutil.copy2(source, destination)
    return destination


def build_final_demo_package(
    *,
    source_dir: Path,
    output_dir: Path,
    repo_root: Path,
    clip_root: Path,
    threshold: float,
    allow_missing: bool = False,
) -> tuple[pd.DataFrame, FinalDemoPackageSummary, dict[str, Path]]:
    """Assemble the Phase 5B final demo package on disk.

    Returns the unified manifest dataframe, the summary dataclass, and a dict
    pointing to the manifest CSV and summary JSON files that were written.

    If any source clip cannot be located on disk, the build fails before any
    files are written. Pass `allow_missing=True` to keep the previous behavior
    of skipping missing clips and recording them in the summary.
    """

    resolved_source_dir = source_dir.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_clip_root = clip_root.expanduser().resolve()

    if not resolved_source_dir.exists():
        raise FileNotFoundError(f"Branch 4C source dir not found: {resolved_source_dir}")

    # Pre-flight pass: load all CSVs and resolve every clip path before we
    # touch the output directory. This way a missing clip fails the build
    # cleanly with no half-written package on disk.
    loaded_frames: dict[str, pd.DataFrame] = {}
    resolved_paths: dict[str, list[Path | None]] = {}
    category_counts: dict[str, int] = {}
    missing_source_files: list[str] = []
    notes: list[str] = []

    for category, file_name in DEFAULT_SOURCE_CSV_NAMES.items():
        frame = _load_category_csv(resolved_source_dir, file_name, category)
        category_counts[category] = int(len(frame))
        loaded_frames[category] = frame
        if frame.empty:
            notes.append(f"Source CSV for '{category}' had zero rows.")
            resolved_paths[category] = []
            continue

        per_row_paths: list[Path | None] = []
        for _, row in frame.iterrows():
            unique_id = str(row.get("unique_id"))
            source_clip = _resolve_source_clip(
                row, repo_root=repo_root, clip_root=resolved_clip_root
            )
            if source_clip is None:
                missing_source_files.append(f"{category}:{unique_id}")
            per_row_paths.append(source_clip)
        resolved_paths[category] = per_row_paths

    if missing_source_files and not allow_missing:
        preview = ", ".join(missing_source_files[:5])
        suffix = "" if len(missing_source_files) <= 5 else f" (+{len(missing_source_files) - 5} more)"
        raise FinalDemoPackageError(
            f"{len(missing_source_files)} source clip(s) could not be located on disk: "
            f"{preview}{suffix}. Re-run with allow_missing=True to skip them."
        )

    # Copy pass: only runs if pre-flight passed (or allow_missing was set).
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_frames: list[pd.DataFrame] = []
    for category, frame in loaded_frames.items():
        if frame.empty:
            continue
        category_subdir = resolved_output_dir / category
        category_subdir.mkdir(parents=True, exist_ok=True)

        copied_records: list[dict[str, object]] = []
        for (_, row), source_clip in zip(frame.iterrows(), resolved_paths[category]):
            if source_clip is None:
                # Only reachable when allow_missing=True.
                continue
            unique_id = str(row.get("unique_id"))
            destination = _copy_clip(source_clip, category_subdir, unique_id)
            record: dict[str, object] = {column: row.get(column) for column in frame.columns}
            record["final_category"] = category
            record["category_rank"] = row.get("category_rank")
            record["source_clip_path"] = str(source_clip)
            # Store packaged path relative to the output dir so the manifest
            # stays valid if the package folder is moved or zipped.
            record["packaged_clip_path"] = str(destination.relative_to(resolved_output_dir))
            copied_records.append(record)

        if copied_records:
            manifest_frames.append(pd.DataFrame.from_records(copied_records))

    if not manifest_frames:
        raise FinalDemoPackageError(
            "No clips were copied. Verify Branch 4C CSVs and clip paths exist."
        )

    manifest = pd.concat(manifest_frames, ignore_index=True)
    leading = [column for column in MANIFEST_COLUMNS if column in manifest.columns]
    trailing = [column for column in manifest.columns if column not in leading]
    manifest = manifest.loc[:, [*leading, *trailing]]

    output_paths: dict[str, Path] = {
        "manifest_csv": resolved_output_dir / "final_demo_manifest.csv",
        "summary_json": resolved_output_dir / "final_demo_package_summary.json",
    }
    manifest.to_csv(output_paths["manifest_csv"], index=False)

    summary = FinalDemoPackageSummary(
        branch="5b_final_demo_package",
        source_dir=str(resolved_source_dir),
        output_dir=str(resolved_output_dir),
        threshold=float(threshold),
        allow_missing=bool(allow_missing),
        category_counts=category_counts,
        total_clips_copied=int(len(manifest)),
        missing_source_files=missing_source_files,
        notes=notes,
    )
    output_paths["summary_json"].write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    return manifest, summary, output_paths
