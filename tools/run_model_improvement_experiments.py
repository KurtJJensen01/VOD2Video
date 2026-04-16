#!/usr/bin/env python3
"""Branch 4B model improvement experiment runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.model_improvement import (  # noqa: E402
    build_default_model_experiments,
    build_feature_setup_source_payload,
    build_model_experiment_summary,
    build_model_experiment_table,
    load_baseline_summary,
    resolve_base_positive_class_weight,
    resolve_feature_columns_for_model_improvement,
    run_model_experiment,
    write_model_experiment_outputs,
)
from vod2video.training_data import TrainingDataError  # noqa: E402


DEFAULT_FEATURE_IMPROVEMENT_SUMMARY = (
    REPO_ROOT / "artifacts" / "feature_improvement" / "branch_4a" / "feature_experiment_summary.json"
)
DEFAULT_BASELINE_SUMMARY = (
    REPO_ROOT / "artifacts" / "training" / "branch_3a_real_baseline" / "metrics_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeatable Branch 4B model-improvement experiments on the current best feature setup."
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
        default=REPO_ROOT / "artifacts" / "model_improvement" / "branch_4b",
        help="Directory where Branch 4B experiment outputs will be written.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Baseline number of epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Baseline optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Baseline hidden layer width.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Baseline dropout rate.")
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
        help="Validation metric used for best-model checkpointing and experiment ranking.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Optional subset of Branch 4B experiment names to run.",
    )
    parser.add_argument(
        "--feature-improvement-summary",
        type=Path,
        default=DEFAULT_FEATURE_IMPROVEMENT_SUMMARY,
        help="Branch 4A summary JSON used to select the current best feature setup.",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_SUMMARY,
        help="Optional Branch 3A summary JSON used for comparison deltas.",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        help="Optional explicit feature column override. If omitted, Branch 4A best features are used.",
    )
    parser.add_argument(
        "--disable-auto-class-weight",
        action="store_true",
        help="Disable automatic positive-class weighting for the baseline and tuned experiments.",
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        help="Explicit positive-class loss weight override for weighted experiments.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_experiments = build_default_model_experiments()
    selected_names = set(args.experiments or [])
    experiments = [
        experiment
        for experiment in all_experiments
        if not selected_names or experiment.name in selected_names
    ]
    if not experiments:
        print("No experiments selected.", file=sys.stderr)
        return 1

    if args.feature_columns:
        feature_columns = [str(name) for name in args.feature_columns]
        feature_setup_source_payload = {
            "type": "explicit_cli_override",
            "feature_count": len(feature_columns),
        }
    else:
        feature_columns, feature_setup_source = resolve_feature_columns_for_model_improvement(
            feature_improvement_summary_path=args.feature_improvement_summary,
            baseline_summary_path=args.baseline_summary,
        )
        feature_setup_source_payload = build_feature_setup_source_payload(feature_setup_source)

    try:
        base_positive_class_weight = resolve_base_positive_class_weight(
            feature_manifest_path=args.feature_manifest,
            explicit_positive_class_weight=args.positive_class_weight,
            disable_auto_class_weight=args.disable_auto_class_weight,
        )
    except (FileNotFoundError, TrainingDataError, ValueError) as exc:
        print(f"Could not resolve positive class weight: {exc}", file=sys.stderr)
        return 1

    baseline_summary = load_baseline_summary(args.baseline_summary)
    results: list[dict[str, object]] = []

    for experiment in experiments:
        run_output_dir = output_dir / "runs" / experiment.name
        print(f"Running model experiment: {experiment.name}")
        try:
            result = run_model_experiment(
                experiment=experiment,
                feature_manifest_path=args.feature_manifest,
                feature_names=feature_columns,
                run_output_dir=run_output_dir,
                batch_size=args.batch_size,
                weight_decay=args.weight_decay,
                seed=args.seed,
                device=args.device,
                monitor_metric=args.monitor_metric,
                base_epochs=args.epochs,
                base_learning_rate=args.learning_rate,
                base_hidden_dim=args.hidden_dim,
                base_dropout=args.dropout,
                base_positive_class_weight=base_positive_class_weight,
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
                    "best_epoch": result["best_epoch"],
                    "val_f1": result["best_epoch_metrics"]["val"]["f1"],
                    "test_f1": result["best_epoch_metrics"]["test"]["f1"],
                    "class_weight_enabled": result["class_weight_enabled"],
                }
            )
        )

    comparison_table = build_model_experiment_table(
        results,
        baseline_summary=baseline_summary,
        rank_metric=f"val_{args.monitor_metric}" if args.monitor_metric != "loss" else "best_metric_value",
    )
    summary = build_model_experiment_summary(
        feature_manifest_path=args.feature_manifest,
        feature_columns=feature_columns,
        feature_setup_source=feature_setup_source_payload,
        results=results,
        comparison_table=comparison_table,
        baseline_summary=baseline_summary,
        output_dir=output_dir,
    )
    output_paths = write_model_experiment_outputs(
        output_dir=output_dir,
        comparison_table=comparison_table,
        summary=summary,
    )

    print("Branch 4B model experiments complete")
    print(f"feature manifest={args.feature_manifest.expanduser().resolve()}")
    print(f"output dir={output_dir}")
    print(f"summary json={output_paths['summary_json']}")
    print(f"table csv={output_paths['table_csv']}")
    if not comparison_table.empty:
        print("best experiment=" + json.dumps(comparison_table.iloc[0].to_dict(), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
