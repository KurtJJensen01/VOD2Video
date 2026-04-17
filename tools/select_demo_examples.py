#!/usr/bin/env python3
"""CLI entry point for the Branch 4C demo example selection workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.demo_selection import (  # noqa: E402
    DemoSelectionError,
    run_demo_selection,
)

DEFAULT_MODEL_SUMMARY = (
    REPO_ROOT / "artifacts" / "model_improvement" / "branch_4b" / "model_experiment_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select ranked demo-worthy examples from Branch 2C/3B/4B prediction artifacts."
    )
    parser.add_argument(
        "--prediction-csv",
        type=Path,
        help="Optional scored or evaluated prediction CSV to use directly.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        help="Optional Branch 3B-style ranked review CSV. If provided, it is used directly.",
    )
    parser.add_argument(
        "--model-summary-json",
        type=Path,
        default=DEFAULT_MODEL_SUMMARY,
        help="Branch 4B summary JSON used to resolve the current best experiment by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "demo_selection" / "branch_4c",
        help="Directory where Branch 4C demo-selection artifacts will be written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used when review_group is not already present.",
    )
    parser.add_argument(
        "--top-k-per-group",
        type=int,
        default=8,
        help="How many ranked examples to keep for each main presentation group.",
    )
    parser.add_argument(
        "--borderline-k",
        type=int,
        default=6,
        help="How many near-threshold examples to keep in the optional borderline output.",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=5,
        help="How many selected examples from each main group to print to stdout.",
    )
    return parser.parse_args()


def _print_group_preview(name: str, frame, show_top: int) -> None:
    if show_top <= 0 or frame.empty:
        return
    preview_columns = [
        column
        for column in (
            "demo_rank",
            "unique_id",
            "label",
            "predicted_probability",
            "predicted_class",
            "resolved_clip_path",
        )
        if column in frame.columns
    ]
    print()
    print(name)
    print(frame.loc[:, preview_columns].head(show_top).to_string(index=False))


def main() -> int:
    args = parse_args()

    try:
        prepared, selections, summary, output_paths = run_demo_selection(
            prediction_csv_path=args.prediction_csv,
            review_csv_path=args.review_csv,
            model_summary_json_path=args.model_summary_json,
            output_dir=args.output_dir,
            threshold=args.threshold,
            top_k_per_group=args.top_k_per_group,
            borderline_k=args.borderline_k,
        )
    except (FileNotFoundError, DemoSelectionError, ValueError) as exc:
        print(f"Demo example selection failed: {exc}", file=sys.stderr)
        return 1

    print("Branch 4C demo selection complete")
    print(f"output dir={summary.output_dir}")
    print(f"source type={summary.source_type}")
    print(f"source prediction csv={summary.source_prediction_csv}")
    if summary.source_experiment_name:
        print(f"source experiment={summary.source_experiment_name}")
    print(f"summary json={output_paths['summary_json']}")
    print(f"top true positives csv={output_paths['top_true_positives_csv']}")
    print(f"top false positives csv={output_paths['top_false_positives_csv']}")
    print(f"top false negatives csv={output_paths['top_false_negatives_csv']}")
    print(f"top ranked highlights csv={output_paths['top_ranked_highlights_csv']}")
    print(f"borderline csv={output_paths['borderline_examples_csv']}")
    print(
        "summary="
        + json.dumps(
            {
                "total_rows": summary.total_rows,
                "labeled_rows": summary.labeled_rows,
                "threshold": summary.threshold,
                "review_group_counts": summary.review_group_counts,
                "selected_group_counts": summary.selected_group_counts,
            }
        )
    )

    if args.show_top > 0:
        for group_name in (
            "top_true_positives",
            "top_false_positives",
            "top_false_negatives",
            "top_ranked_highlights",
        ):
            _print_group_preview(group_name, selections[group_name], args.show_top)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
