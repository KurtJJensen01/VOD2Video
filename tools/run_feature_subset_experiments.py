#!/usr/bin/env python3
"""Branch 4A feature subset experiment runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.feature_improvement import (  # noqa: E402
    build_default_experiments,
    build_feature_experiment_summary,
    build_feature_experiment_table,
    build_feature_groups,
    load_baseline_summary,
    resolve_positive_class_weight,
    run_feature_experiment,
    write_feature_experiment_outputs,
)
from vod2video.training_data import TrainingDataError  # noqa: E402


DEFAULT_BASELINE_SUMMARY = (
    REPO_ROOT / "artifacts" / "training" / "branch_3a_real_baseline" / "metrics_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeatable Branch 4A feature subset experiments on the real feature manifest."
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=REPO_ROOT / "artifacts" / "features" / "branch_2a" / "clip_features.csv",
        help="Path to the Phase 2A feature manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "feature_improvement" / "branch_4a",
        help="Directory where Branch 4A experiment outputs will be written.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs per experiment.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer width.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="f1",
        choices=("loss", "accuracy", "precision", "recall", "f1"),
        help="Validation metric used for best-model checkpointing.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Optional subset of experiment names to run. Defaults to the full Branch 4A set.",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_SUMMARY,
        help="Optional Branch 3A summary JSON used for deltas in the comparison table.",
    )
    parser.add_argument(
        "--disable-auto-class-weight",
        action="store_true",
        help="Disable automatic positive-class weighting from the train split.",
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        help="Explicit positive-class loss weight override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_groups = build_feature_groups()
    all_experiments = build_default_experiments()
    selected_names = set(args.experiments or [])
    experiments = [
        experiment
        for experiment in all_experiments
        if not selected_names or experiment.name in selected_names
    ]
    if not experiments:
        print("No experiments selected.", file=sys.stderr)
        return 1

    try:
        if args.positive_class_weight is not None:
            positive_class_weight = float(args.positive_class_weight)
        elif args.disable_auto_class_weight:
            positive_class_weight = None
        else:
            positive_class_weight = resolve_positive_class_weight(args.feature_manifest)
    except (FileNotFoundError, TrainingDataError, ValueError) as exc:
        print(f"Could not resolve positive class weight: {exc}", file=sys.stderr)
        return 1

    baseline_summary = load_baseline_summary(args.baseline_summary)
    results: list[dict[str, object]] = []

    for experiment in experiments:
        run_output_dir = output_dir / "runs" / experiment.name
        print(f"Running feature experiment: {experiment.name}")
        try:
            result = run_feature_experiment(
                experiment=experiment,
                feature_groups=feature_groups,
                feature_manifest_path=args.feature_manifest,
                run_output_dir=run_output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                seed=args.seed,
                device=args.device,
                monitor_metric=args.monitor_metric,
                positive_class_weight=positive_class_weight,
            )
        except (FileNotFoundError, TrainingDataError, ValueError) as exc:
            print(f"Experiment '{experiment.name}' failed: {exc}", file=sys.stderr)
            return 1

        results.append(result)
        print(
            "completed="
            + json.dumps(
                {
                    "experiment_name": result["experiment_name"],
                    "feature_count": result["feature_count"],
                    "best_epoch": result["best_epoch"],
                    "val_f1": result["best_epoch_metrics"]["val"]["f1"],
                    "test_f1": result["best_epoch_metrics"]["test"]["f1"],
                }
            )
        )

    comparison_table = build_feature_experiment_table(
        results,
        baseline_summary=baseline_summary,
        rank_metric=f"val_{args.monitor_metric}" if args.monitor_metric != "loss" else "best_metric_value",
    )
    summary = build_feature_experiment_summary(
        feature_manifest_path=args.feature_manifest,
        feature_groups=feature_groups,
        results=results,
        comparison_table=comparison_table,
        baseline_summary=baseline_summary,
        output_dir=output_dir,
    )
    output_paths = write_feature_experiment_outputs(
        output_dir=output_dir,
        comparison_table=comparison_table,
        summary=summary,
    )

    print("Branch 4A feature experiments complete")
    print(f"feature manifest={args.feature_manifest.expanduser().resolve()}")
    print(f"output dir={output_dir}")
    print(f"summary json={output_paths['summary_json']}")
    print(f"table csv={output_paths['table_csv']}")
    if not comparison_table.empty:
        print("best experiment=" + json.dumps(comparison_table.iloc[0].to_dict(), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
