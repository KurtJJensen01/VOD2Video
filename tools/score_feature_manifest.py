#!/usr/bin/env python3
"""CLI entry point for the Branch 2C inference/demo flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.inference import InferenceError, score_feature_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a feature manifest with a trained checkpoint and write ranked demo outputs."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "training" / "branch_2b_real_feature_audio_smoke" / "best_model.pt",
        help="Path to a Phase 2B model checkpoint.",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=REPO_ROOT / "artifacts" / "features" / "branch_2a" / "clip_features.csv",
        help="Path to a Phase 2A feature manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "inference" / "branch_2c",
        help="Directory where scored CSVs and summary JSON will be written.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of highest-score clips to write separately.")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional override for the checkpoint decision threshold. Defaults to the saved training threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string to use for scoring, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=5,
        help="How many top-ranked predictions to print to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        predictions, summary, output_paths = score_feature_manifest(
            checkpoint_path=args.checkpoint,
            feature_manifest_path=args.feature_manifest,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            threshold=args.threshold,
            top_k=args.top_k,
            device=args.device,
        )
    except (FileNotFoundError, InferenceError, ValueError, RuntimeError) as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    print("Inference complete")
    print(f"checkpoint={summary.checkpoint_path}")
    print(f"feature manifest={summary.feature_manifest_path}")
    print(f"output scored csv={output_paths['scored_csv']}")
    print(f"output top csv={output_paths['top_csv']}")
    print(f"output summary json={output_paths['summary_json']}")
    print(
        "summary="
        + json.dumps(
            {
                "scored_rows": summary.scored_rows,
                "predicted_positive_count": summary.predicted_positive_count,
                "threshold": summary.threshold,
                "device": summary.device,
                "normalization_applied": summary.normalization_applied,
                "score_min": summary.score_min,
                "score_max": summary.score_max,
                "score_mean": summary.score_mean,
            }
        )
    )
    if args.show_top > 0:
        preview_columns = [
            column
            for column in (
                "score_rank",
                "unique_id",
                "vod_id",
                "segment_id",
                "label",
                "predicted_probability",
                "predicted_class",
            )
            if column in predictions.columns
        ]
        print()
        print(predictions.loc[:, preview_columns].head(args.show_top).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
