#!/usr/bin/env python3
"""CLI entry point for the Branch 3B prediction review pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.prediction_review import (  # noqa: E402
    PredictionReviewError,
    review_prediction_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize scored clip predictions into Branch 3B review artifacts."
    )
    parser.add_argument(
        "--prediction-csv",
        type=Path,
        default=REPO_ROOT / "artifacts" / "training" / "branch_3a_real_baseline" / "test_predictions.csv",
        help="Path to a scored prediction CSV from Branch 2C or Branch 3A.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        help="Optional labels CSV to merge in when the prediction CSV is unlabeled.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "review" / "branch_3b",
        help="Directory where review CSVs and summary JSON will be written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used when predicted_class is not already present.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top-ranked rows to write for presentation review files.",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=5,
        help="How many top-ranked rows to print to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        prepared, summary, output_paths = review_prediction_csv(
            prediction_csv_path=args.prediction_csv,
            output_dir=args.output_dir,
            labels_csv_path=args.labels_csv,
            threshold=args.threshold,
            top_k=args.top_k,
        )
    except (FileNotFoundError, PredictionReviewError, ValueError) as exc:
        print(f"Prediction review failed: {exc}", file=sys.stderr)
        return 1

    print("Prediction review complete")
    print(f"prediction csv={summary.prediction_csv_path}")
    print(f"output dir={summary.output_dir}")
    if args.labels_csv is not None:
        print(f"labels csv={args.labels_csv.expanduser().resolve()}")
    if output_paths is not None:
        for key, path in output_paths.items():
            print(f"{key}={path}")
    print(
        "summary="
        + json.dumps(
            {
                "total_rows": summary.total_rows,
                "labeled_rows": summary.labeled_rows,
                "unlabeled_rows": summary.unlabeled_rows,
                "threshold": summary.threshold,
                "top_k": summary.top_k,
                "review_group_counts": summary.review_group_counts,
                "top_k_summary": summary.top_k_summary,
                "notes": summary.notes,
            }
        )
    )

    if args.show_top > 0:
        preview_columns = [
            column
            for column in (
                "score_rank",
                "review_group",
                "unique_id",
                "vod_id",
                "segment_id",
                "label",
                "predicted_probability",
                "predicted_class",
            )
            if column in prepared.columns
        ]
        print()
        print(prepared.loc[:, preview_columns].head(args.show_top).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
