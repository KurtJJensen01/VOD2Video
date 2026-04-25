#!/usr/bin/env python3
"""CLI entry point for the Phase 1C dataset split plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video import (  # noqa: E402
    DatasetSplitError,
    DatasetValidationError,
    LabeledDatasetSource,
    SplitConfig,
    discover_labeled_dataset_sources,
    format_split_summaries,
    format_summary,
    load_labeled_dataset,
    split_labeled_dataset,
    write_split_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load labeled VOD clips and build a repeatable train/val/test split."
    )
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
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic block assignment (default: 42).",
    )
    parser.add_argument(
        "--neighbor-window-seconds",
        type=int,
        default=60,
        help=(
            "Nearby labeled clips within this many seconds stay in the same split block "
            "(default: 60)."
        ),
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of rows to target for train (default: 0.7).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of rows to target for validation (default: 0.15).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of rows to target for test (default: 0.15).",
    )
    parser.add_argument(
        "--show-head",
        type=int,
        default=5,
        help="Number of split rows to print from the combined manifest (default: 5).",
    )
    parser.add_argument(
        "--write-dir",
        type=Path,
        help=(
            "Optional output directory for generated train.csv / val.csv / test.csv "
            "manifests. These outputs are convenience artifacts, not required source files."
        ),
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
    config = SplitConfig(
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        neighbor_window_seconds=args.neighbor_window_seconds,
    )

    try:
        sources = build_sources_from_args(args, repo_root=REPO_ROOT)
        loaded = load_labeled_dataset(sources)
        dataset_split = split_labeled_dataset(loaded.dataframe, config=config)
    except (DatasetSplitError, DatasetValidationError, FileNotFoundError, ValueError) as exc:
        print(f"Dataset split failed: {exc}", file=sys.stderr)
        return 1

    print(format_summary(loaded.summary))
    print()
    print(
        "Split config: "
        f"train={config.train_fraction:.2f}, val={config.val_fraction:.2f}, "
        f"test={config.test_fraction:.2f}, seed={config.seed}, "
        f"neighbor_window_seconds={config.neighbor_window_seconds}"
    )
    print(format_split_summaries(dataset_split.summaries))

    if args.show_head > 0:
        preview_columns = [
            "unique_id",
            "vod_id",
            "segment_id",
            "label",
            "start_time_seconds",
            config.block_column,
            config.split_column,
        ]
        available_columns = [
            column for column in preview_columns if column in dataset_split.dataframe.columns
        ]
        print()
        print(f"First {min(args.show_head, len(dataset_split.dataframe))} rows:")
        print(dataset_split.dataframe[available_columns].head(args.show_head).to_string(index=False))

    if args.write_dir is not None:
        manifest_paths = write_split_manifests(dataset_split, args.write_dir)
        print()
        print("Wrote split manifests:")
        for key, path in manifest_paths.items():
            print(f"  {key}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
