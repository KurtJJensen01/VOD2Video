#!/usr/bin/env python3
"""CLI entry point for the Phase 7 highlight clip selection workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.highlight_selection import (  # noqa: E402
    HighlightSelectionError,
    select_highlight_clips,
)

DEFAULT_INPUT_CSV = (
    REPO_ROOT / "artifacts" / "inference" / "phase_6" / "inference" / "predictions.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "highlight_selection" / "phase_7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7 highlight clip selection. Reads Phase 6's ranked predictions "
            "and produces a curated set of clips ready for Phase 8 video assembly."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Phase 6 ranked-prediction CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where Phase 7 selection outputs are written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold; clips below this are dropped.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Maximum number of clips to keep after the threshold filter.",
    )
    parser.add_argument(
        "--min-gap-seconds",
        type=float,
        default=0.0,
        help=(
            "Redundancy filter: minimum gap (seconds) between selected clips "
            "within a VOD. 0 disables the filter. Larger values drop adjacent "
            "or near-adjacent clips, keeping only the higher-scored one."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest, summary, output_paths = select_highlight_clips(
            input_csv=args.input,
            output_dir=args.output_dir,
            threshold=args.threshold,
            top_k=args.top_k,
            min_gap_seconds=args.min_gap_seconds,
        )
    except (FileNotFoundError, HighlightSelectionError, ValueError) as exc:
        print(f"Phase 7 highlight selection failed: {exc}", file=sys.stderr)
        return 1

    print("Phase 7 highlight selection complete")
    print(f"output dir={summary.output_dir}")
    print(f"manifest csv={output_paths['manifest_csv']}")
    print(f"summary json={output_paths['summary_json']}")
    print(
        "stage counts="
        + json.dumps(
            {
                "input": summary.total_input_rows,
                "after_threshold": summary.rows_after_threshold,
                "after_top_k": summary.rows_after_top_k,
                "after_redundancy": summary.rows_after_redundancy,
            }
        )
    )
    print(f"total clips copied={summary.total_clips_copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
