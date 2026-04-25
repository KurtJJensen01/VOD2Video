#!/usr/bin/env python3
"""CLI entry point for the Phase 1A dataset loader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.dataset_loader import (
    DatasetValidationError,
    LabeledDatasetSource,
    discover_labeled_dataset_sources,
    format_summary,
    load_labeled_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and validate labeled VOD clip datasets.")
    parser.add_argument(
        "--csv",
        action="append",
        dest="csv_paths",
        help="Path to a labeled CSV file. Repeat once per source.",
    )
    parser.add_argument(
        "--clip-root",
        action="append",
        dest="clip_roots",
        help=(
            "Root directory used to resolve each source's relative clip_path values. "
            "Repeat once per source, in the same order as --csv."
        ),
    )
    parser.add_argument(
        "--show-head",
        type=int,
        default=5,
        help="Number of validated rows to print after loading (default: 5).",
    )
    return parser.parse_args()


def build_sources_from_args(args: argparse.Namespace, repo_root: Path) -> list[LabeledDatasetSource]:
    if not args.csv_paths and not args.clip_roots:
        return discover_labeled_dataset_sources(repo_root / "labeling_test")

    csv_paths = args.csv_paths or []
    clip_roots = args.clip_roots or []

    if len(csv_paths) != len(clip_roots):
        raise ValueError("You must provide the same number of --csv and --clip-root arguments.")

    return [
        LabeledDatasetSource(csv_path=Path(csv_path), clip_root=Path(clip_root))
        for csv_path, clip_root in zip(csv_paths, clip_roots, strict=True)
    ]


def main() -> int:
    args = parse_args()

    try:
        sources = build_sources_from_args(args, repo_root=REPO_ROOT)
        loaded = load_labeled_dataset(sources)
    except (DatasetValidationError, FileNotFoundError, ValueError) as exc:
        print(f"Dataset loader failed: {exc}", file=sys.stderr)
        return 1

    print(format_summary(loaded.summary))
    print()
    print("Validated columns:")
    print(", ".join(loaded.dataframe.columns))

    if args.show_head > 0:
        preview_columns = ["unique_id", "vod_id", "segment_id", "label", "clip_path", "resolved_clip_path"]
        print()
        print(f"First {min(args.show_head, len(loaded.dataframe))} rows:")
        print(loaded.dataframe[preview_columns].head(args.show_head).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
