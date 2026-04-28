"""Highlight clip selection for Phase 7.

Reads a Phase 6 ranked-prediction CSV and applies three filters in order:

1. **Threshold** — drop clips below `threshold`.
2. **Top-K cap** — keep at most `top_k` clips after the threshold filter.
3. **Redundancy** — within a single VOD, drop clips whose interval is closer
   than `min_gap_seconds` to an already-selected (higher-scored) clip. This
   is non-maximum suppression in the time domain. `min_gap_seconds=0` disables
   the filter entirely.

The output is a self-contained folder ready for Phase 8 video assembly:
copied clip files, a chronological manifest CSV, and a selection summary JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import shutil
from pathlib import Path

import pandas as pd

# Columns required in the Phase 6 ranked-prediction CSV.
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "unique_id",
    "vod_id",
    "predicted_probability",
    "start_time_seconds",
    "end_time_seconds",
)

# Manifest column order. Any extra columns inherited from the Phase 6 CSV are
# appended after these for traceability.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "selection_rank",
    "unique_id",
    "vod_id",
    "segment_id",
    "predicted_probability",
    "predicted_class",
    "label",
    "start_time_seconds",
    "end_time_seconds",
    "source_clip_path",
    "packaged_clip_path",
)


class HighlightSelectionError(ValueError):
    """Raised when Phase 7 selection inputs are invalid or unbuildable."""


@dataclass(frozen=True)
class HighlightSelectionSummary:
    branch: str
    input_csv: str
    output_dir: str
    threshold: float
    top_k: int
    min_gap_seconds: float
    total_input_rows: int
    rows_after_threshold: int
    rows_after_top_k: int
    rows_after_redundancy: int
    total_clips_copied: int
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _resolve_clip_path(row: dict, input_csv_dir: Path) -> Path | None:
    """Locate the on-disk clip file for one row.

    Tries `resolved_clip_path` first (absolute path written by Phase 6), then
    falls back to `clip_path` resolved against the parent of the input CSV
    directory (Phase 6 stores clips one level above its predictions CSV) and
    against the input CSV directory itself.
    """

    candidates: list[Path] = []
    resolved = row.get("resolved_clip_path")
    if isinstance(resolved, str) and resolved:
        candidates.append(Path(resolved))
    relative = row.get("clip_path")
    if isinstance(relative, str) and relative:
        normalized = relative.replace("\\", "/")
        candidates.append(input_csv_dir.parent / normalized)
        candidates.append(input_csv_dir / normalized)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _apply_threshold(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    probabilities = pd.to_numeric(frame["predicted_probability"], errors="coerce")
    if probabilities.isna().any():
        raise HighlightSelectionError(
            "predicted_probability must be numeric for every input row."
        )
    return frame.loc[probabilities >= float(threshold)].copy()


def _apply_top_k(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    top_k_count = max(int(top_k), 0)
    if top_k_count == 0 or frame.empty:
        return frame.iloc[0:0].copy()
    return (
        frame.sort_values(by="predicted_probability", ascending=False, kind="mergesort")
        .head(top_k_count)
        .reset_index(drop=True)
    )


def _apply_redundancy_filter(frame: pd.DataFrame, min_gap_seconds: float) -> pd.DataFrame:
    """Time-domain non-maximum suppression, applied per VOD.

    Clips are considered in descending probability order. A candidate is
    accepted unless an already-accepted clip (within the same VOD) sits closer
    than `min_gap_seconds` to it. `min_gap_seconds <= 0` short-circuits to
    "no filter" and returns the input unchanged.
    """

    if min_gap_seconds <= 0 or frame.empty:
        return frame.reset_index(drop=True)

    sorted_indices = frame.sort_values(
        by="predicted_probability", ascending=False, kind="mergesort"
    ).index.tolist()

    accepted_indices: list[int] = []
    for idx in sorted_indices:
        candidate = frame.loc[idx]
        candidate_start = float(candidate["start_time_seconds"])
        candidate_end = float(candidate["end_time_seconds"])
        candidate_vod = candidate["vod_id"]

        conflict = False
        for accepted_idx in accepted_indices:
            accepted = frame.loc[accepted_idx]
            if accepted["vod_id"] != candidate_vod:
                continue
            accepted_start = float(accepted["start_time_seconds"])
            accepted_end = float(accepted["end_time_seconds"])
            # Distance between two intervals: positive when disjoint, zero
            # when touching, negative when overlapping. Reject when it is less
            # than the configured minimum.
            gap = max(candidate_start - accepted_end, accepted_start - candidate_end)
            if gap < float(min_gap_seconds):
                conflict = True
                break

        if not conflict:
            accepted_indices.append(idx)

    return frame.loc[accepted_indices].reset_index(drop=True)


def select_highlight_clips(
    *,
    input_csv: Path,
    output_dir: Path,
    threshold: float,
    top_k: int,
    min_gap_seconds: float,
) -> tuple[pd.DataFrame, HighlightSelectionSummary, dict[str, Path]]:
    """Run Phase 7 selection end-to-end.

    Returns the manifest dataframe, the summary dataclass, and a dict of the
    output file paths that were written.
    """

    resolved_input = input_csv.expanduser().resolve()
    resolved_output = output_dir.expanduser().resolve()

    if not resolved_input.exists():
        raise FileNotFoundError(
            f"Phase 6 ranked-prediction CSV not found: {resolved_input}"
        )

    raw = pd.read_csv(resolved_input)
    missing_columns = [c for c in REQUIRED_INPUT_COLUMNS if c not in raw.columns]
    if missing_columns:
        raise HighlightSelectionError(
            f"Phase 6 CSV missing required column(s): {', '.join(missing_columns)}"
        )

    notes: list[str] = []
    total_input = int(len(raw))

    threshold_filtered = _apply_threshold(raw, threshold)
    after_threshold = int(len(threshold_filtered))

    top_k_filtered = _apply_top_k(threshold_filtered, top_k)
    after_top_k = int(len(top_k_filtered))
    if int(top_k) <= 0:
        notes.append("top_k <= 0 means no clips selected.")

    redundancy_filtered = _apply_redundancy_filter(top_k_filtered, min_gap_seconds)
    after_redundancy = int(len(redundancy_filtered))

    if redundancy_filtered.empty:
        notes.append("No clips passed the selection filters.")
    else:
        # Final selection is presented in chronological order so it can flow
        # straight into Phase 8 without further sorting.
        redundancy_filtered = redundancy_filtered.sort_values(
            by=["vod_id", "start_time_seconds"], kind="mergesort"
        ).reset_index(drop=True)

    resolved_output.mkdir(parents=True, exist_ok=True)
    clips_subdir = resolved_output / "selected_clips"
    clips_subdir.mkdir(parents=True, exist_ok=True)

    input_csv_dir = resolved_input.parent
    missing_source_files: list[str] = []
    manifest_records: list[dict[str, object]] = []
    for selection_rank, row in enumerate(
        redundancy_filtered.to_dict(orient="records"), start=1
    ):
        unique_id = str(row.get("unique_id"))
        source_path = _resolve_clip_path(row, input_csv_dir=input_csv_dir)
        if source_path is None:
            missing_source_files.append(unique_id)
            continue
        # Prefix with unique_id so clips from different VODs that share a
        # segment_id never collide inside the selected_clips folder.
        destination = clips_subdir / f"{unique_id}__{source_path.name}"
        shutil.copy2(source_path, destination)
        record = dict(row)
        record["selection_rank"] = selection_rank
        record["source_clip_path"] = str(source_path)
        record["packaged_clip_path"] = str(destination.relative_to(resolved_output))
        manifest_records.append(record)

    if missing_source_files:
        raise HighlightSelectionError(
            f"{len(missing_source_files)} selected clip(s) could not be located on disk: "
            f"{', '.join(missing_source_files[:5])}"
            + ("" if len(missing_source_files) <= 5 else f" (+{len(missing_source_files) - 5} more)")
        )

    if manifest_records:
        manifest = pd.DataFrame.from_records(manifest_records)
        leading = [c for c in MANIFEST_COLUMNS if c in manifest.columns]
        trailing = [c for c in manifest.columns if c not in leading]
        manifest = manifest.loc[:, [*leading, *trailing]]
    else:
        manifest = pd.DataFrame(columns=list(MANIFEST_COLUMNS))

    output_paths: dict[str, Path] = {
        "manifest_csv": resolved_output / "selected_highlights_manifest.csv",
        "summary_json": resolved_output / "selection_summary.json",
    }
    manifest.to_csv(output_paths["manifest_csv"], index=False)

    summary = HighlightSelectionSummary(
        branch="7_highlight_selection",
        input_csv=str(resolved_input),
        output_dir=str(resolved_output),
        threshold=float(threshold),
        top_k=int(max(int(top_k), 0)),
        min_gap_seconds=float(min_gap_seconds),
        total_input_rows=total_input,
        rows_after_threshold=after_threshold,
        rows_after_top_k=after_top_k,
        rows_after_redundancy=after_redundancy,
        total_clips_copied=int(len(manifest_records)),
        notes=notes,
    )
    output_paths["summary_json"].write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )

    return manifest, summary, output_paths
