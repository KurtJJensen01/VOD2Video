#!/usr/bin/env python3
"""Phase 6: End-to-end inference pipeline for a new unlabeled VOD.

Usage:
  python tools/run_vod_inference_pipeline.py --input path/to/vod.mp4 --vod-id 3

Output layout under --output-dir:
  clips/                 5-second MP4 files
  clip_manifest.csv      per-clip metadata used by feature extractor
  features/
    clip_features.csv
    feature_summary.json
  inference/
    scored_clips.csv     all clips ranked by predicted_probability
    top_highlights.csv   top --top-k clips only
    inference_summary.json
  pipeline_summary.json  high-level run summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.clip_features import (  # noqa: E402
    AudioExtractionConfig,
    ClipFeatureExtractionError,
    ClipSamplingConfig,
    build_feature_manifest,
    build_feature_manifest_summary,
    load_feature_source_manifest,
    resolve_audio_tool_status,
    write_feature_manifest_outputs,
)
from vod2video.inference import InferenceError, score_feature_manifest  # noqa: E402

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "model_improvement"
    / "branch_4b"
    / "runs"
    / "lower_learning_rate"
    / "best_model.pt"
)

_MANIFEST_FIELDNAMES = [
    "vod_id",
    "segment_id",
    "clip_name",
    "clip_path",
    "start_time_seconds",
    "end_time_seconds",
    "start_time_hhmmss",
    "end_time_hhmmss",
    "label",
    "split",
    "unique_id",
    "resolved_clip_path",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _require_binary(name: str, path_override: str | None) -> str:
    if path_override:
        candidate = Path(path_override).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        raise RuntimeError(f"Binary not found at specified path: {path_override}")
    found = shutil.which(name)
    if found is None:
        raise RuntimeError(f"Required binary not found in PATH: {name}")
    return found


def _probe_duration(video_path: Path, ffprobe: str) -> float:
    result = _run([
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ])
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not parse duration: {result.stdout!r}") from exc


def _hhmmss(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _export_clip(
    input_video: Path,
    output_clip: Path,
    start: float,
    duration: float,
    ffmpeg: str,
) -> None:
    result = _run([
        ffmpeg, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_video),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(output_clip),
    ])
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output_clip.name}:\n{result.stderr}")


# ---------------------------------------------------------------------------
# Step 1: clip generation
# ---------------------------------------------------------------------------


def generate_clips(
    input_video: Path,
    clips_dir: Path,
    vod_id: str,
    segment_length: int,
    ffmpeg: str,
    ffprobe: str,
) -> list[dict]:
    duration = _probe_duration(input_video, ffprobe)
    segment_count = math.ceil(duration / segment_length)
    print(f"  Duration: {duration:.1f}s  Clips: {segment_count}")

    rows: list[dict] = []
    for index in range(segment_count):
        start = index * segment_length
        end = min((index + 1) * segment_length, duration)
        clip_duration = end - start

        segment_id = f"seg_{index + 1:05d}"
        clip_name = f"{segment_id}_{int(start):06d}_{int(end):06d}.mp4"
        output_clip = clips_dir / clip_name

        if not output_clip.exists():
            _export_clip(input_video, output_clip, start, clip_duration, ffmpeg)

        if (index + 1) % 200 == 0 or (index + 1) == segment_count:
            print(f"  [{index + 1}/{segment_count}] {clip_name}")

        rows.append({
            "vod_id": vod_id,
            "segment_id": segment_id,
            "clip_name": clip_name,
            "clip_path": str(Path("clips") / clip_name),
            "start_time_seconds": f"{start:.3f}",
            "end_time_seconds": f"{end:.3f}",
            "start_time_hhmmss": _hhmmss(start),
            "end_time_hhmmss": _hhmmss(end),
            # label and split are placeholder values required by the feature extractor schema.
            # The model's actual highlight predictions are in predicted_class / predicted_probability.
            "label": "0",
            "split": "infer",
            "unique_id": f"{vod_id}_{segment_id}",
            "resolved_clip_path": str(output_clip.resolve()),
        })

    return rows


def write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6: Run the full inference pipeline on a new unlabeled VOD."
    )
    parser.add_argument("--input", required=True, help="Path to input VOD file")
    parser.add_argument(
        "--vod-id",
        required=True,
        help="Short identifier for this VOD (e.g. '3'); used in unique_id values",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "inference" / "phase_6",
        help="Root output directory (default: artifacts/inference/phase_6)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint (default: lower_learning_rate from Phase 4B)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for positive class (0.0-1.0). Defaults to the value saved in the checkpoint.",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=5,
        help="Clip length in seconds (default: 5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top-ranked clips to write to top_highlights.csv (default: 20)",
    )
    parser.add_argument("--ffmpeg-path", help="Explicit path to ffmpeg if not on PATH")
    parser.add_argument("--ffprobe-path", help="Explicit path to ffprobe if not on PATH")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_video = Path(args.input).expanduser().resolve()
    if not input_video.exists():
        print(f"Input file not found: {input_video}", file=sys.stderr)
        return 1

    try:
        ffmpeg = _require_binary("ffmpeg", args.ffmpeg_path)
        ffprobe = _require_binary("ffprobe", args.ffprobe_path)
    except RuntimeError as exc:
        print(f"Tool error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    clips_dir = output_dir / "clips"
    features_dir = output_dir / "features"
    inference_dir = output_dir / "inference"
    for d in (clips_dir, features_dir, inference_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: generate clips
    # ------------------------------------------------------------------
    print(f"\n[Step 1] Generating {args.segment_length}s clips from: {input_video.name}")
    try:
        clip_rows = generate_clips(
            input_video=input_video,
            clips_dir=clips_dir,
            vod_id=args.vod_id,
            segment_length=args.segment_length,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
    except RuntimeError as exc:
        print(f"Clip generation failed: {exc}", file=sys.stderr)
        return 1

    manifest_path = output_dir / "clip_manifest.csv"
    write_manifest(manifest_path, clip_rows)
    print(f"  Manifest: {manifest_path}  ({len(clip_rows)} clips)")

    # ------------------------------------------------------------------
    # Step 2: extract features
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Extracting features from {len(clip_rows)} clips...")
    audio_config = AudioExtractionConfig(
        enabled=True,
        ffmpeg_path=args.ffmpeg_path,
        ffprobe_path=args.ffprobe_path,
    )
    sampling_config = ClipSamplingConfig()
    audio_tool_status = resolve_audio_tool_status(audio_config)
    if audio_tool_status.warning:
        print(f"  Warning: {audio_tool_status.warning}")

    try:
        source_manifest = load_feature_source_manifest(manifest_path)
        feature_manifest = build_feature_manifest(
            source_manifest,
            sampling_config=sampling_config,
            audio_config=audio_config,
            include_optional_columns=["start_time_seconds", "end_time_seconds"],
        )
        feature_summary = build_feature_manifest_summary(
            feature_manifest,
            sampling_config=sampling_config,
            audio_config=audio_config,
            audio_tool_status=audio_tool_status,
        )
        feature_paths = write_feature_manifest_outputs(
            feature_manifest,
            output_dir=features_dir,
            summary=feature_summary,
        )
    except (ClipFeatureExtractionError, FileNotFoundError, ValueError) as exc:
        print(f"Feature extraction failed: {exc}", file=sys.stderr)
        return 1

    features_csv = feature_paths["features_csv"]
    print(f"  Feature columns: {len(feature_summary.feature_columns)}")

    # The checkpoint was trained with three derived columns that training_data.py
    # computes at training time but the feature extractor does not produce.
    # Add them now to match the checkpoint's expected feature_names exactly.
    feat_df = pd.read_csv(features_csv)
    feat_df["duration_seconds"] = (
        pd.to_numeric(feat_df["end_time_seconds"]) - pd.to_numeric(feat_df["start_time_seconds"])
    )
    feat_df["segment_index"] = pd.to_numeric(
        feat_df["segment_id"].astype("string").str.extract(r"(\d+)$", expand=False),
        errors="coerce",
    )
    vod_values = feat_df["vod_id"].astype("string").tolist()
    unique_vods = {v: i for i, v in enumerate(dict.fromkeys(vod_values))}
    feat_df["vod_index"] = [float(unique_vods[v]) for v in vod_values]
    feat_df.to_csv(features_csv, index=False)
    print(f"  Features CSV: {features_csv}  (+duration_seconds, segment_index, vod_index)")

    # ------------------------------------------------------------------
    # Step 3: score and rank clips
    # ------------------------------------------------------------------
    threshold_note = f"{args.threshold}" if args.threshold is not None else "from checkpoint"
    print(f"\n[Step 3] Scoring clips with: {args.checkpoint.name}  (threshold: {threshold_note})")
    try:
        predictions, inf_summary, inf_paths = score_feature_manifest(
            checkpoint_path=args.checkpoint,
            feature_manifest_path=features_csv,
            output_dir=inference_dir,
            top_k=args.top_k,
            threshold=args.threshold,
        )
    except (FileNotFoundError, InferenceError, ValueError, RuntimeError) as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    print(f"  Scored: {inf_summary.scored_rows} clips")
    print(f"  Predicted highlights: {inf_summary.predicted_positive_count}")
    print(f"  Score range: [{inf_summary.score_min:.3f}, {inf_summary.score_max:.3f}]")
    print(f"  Ranked CSV: {inf_paths['scored_csv']}")
    print(f"  Top {args.top_k} CSV: {inf_paths['top_csv']}")

    preview_cols = [
        c for c in (
            "score_rank", "unique_id", "start_time_seconds",
            "end_time_seconds", "predicted_probability", "predicted_class",
        )
        if c in predictions.columns
    ]
    print(f"\nTop {min(args.top_k, 10)} predicted highlights:")
    print(predictions[preview_cols].head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # Pipeline summary
    # ------------------------------------------------------------------
    pipeline_summary = {
        "vod_id": args.vod_id,
        "input_video": str(input_video),
        "segment_length_seconds": args.segment_length,
        "total_clips": len(clip_rows),
        "checkpoint": str(args.checkpoint),
        "predicted_highlights": inf_summary.predicted_positive_count,
        "threshold": inf_summary.threshold,
        "score_min": inf_summary.score_min,
        "score_max": inf_summary.score_max,
        "score_mean": inf_summary.score_mean,
        "outputs": {
            "clip_manifest": str(manifest_path),
            "features_csv": str(features_csv),
            "ranked_csv": str(inf_paths["scored_csv"]),
            "top_highlights_csv": str(inf_paths["top_csv"]),
        },
    }
    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(pipeline_summary, indent=2), encoding="utf-8")
    print(f"\nDone. Pipeline summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
