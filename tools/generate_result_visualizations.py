#!/usr/bin/env python3
"""CLI entry point for Branch 3C result visualization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.visualization import (  # noqa: E402
    VisualizationError,
    generate_visualization_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Branch 3C presentation-ready visualizations from saved 3A/3B artifacts."
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "training" / "branch_3a_real_baseline",
        help="Directory containing Branch 3A metric and history artifacts.",
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "review" / "branch_3b",
        help="Review directory or review_summary.json path from Branch 3B.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("val", "test"),
        help="Which evaluated split to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "visualization" / "branch_3c",
        help="Directory where Branch 3C outputs will be written.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV/JSON artifacts only, without PNG plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        artifacts = generate_visualization_artifacts(
            training_dir=args.training_dir,
            review_dir=args.review_dir,
            output_dir=args.output_dir,
            split_name=args.split,
            generate_plots=not args.skip_plots,
        )
    except (FileNotFoundError, VisualizationError, ValueError) as exc:
        print(f"Visualization generation failed: {exc}", file=sys.stderr)
        return 1

    print("Branch 3C visualization generation complete")
    print(f"split={artifacts.split_name}")
    print(f"output dir={artifacts.output_dir}")
    print(f"metrics table={artifacts.metrics_table_csv}")
    print(f"confusion matrix={artifacts.confusion_matrix_csv}")
    if artifacts.metrics_comparison_csv is not None:
        print(f"metrics comparison={artifacts.metrics_comparison_csv}")
    if artifacts.review_summary_table_csv is not None:
        print(f"review summary table={artifacts.review_summary_table_csv}")
    if artifacts.epoch_metrics_csv is not None:
        print(f"epoch metrics={artifacts.epoch_metrics_csv}")
    if artifacts.confusion_matrix_png is not None:
        print(f"confusion matrix png={artifacts.confusion_matrix_png}")
    if artifacts.metrics_table_png is not None:
        print(f"metrics table png={artifacts.metrics_table_png}")
    if artifacts.metric_bar_chart_png is not None:
        print(f"metric bar chart png={artifacts.metric_bar_chart_png}")
    if artifacts.epoch_overview_png is not None:
        print(f"epoch overview png={artifacts.epoch_overview_png}")
    print(f"summary={artifacts.visualization_summary_json}")
    print("artifacts=" + json.dumps(artifacts.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
