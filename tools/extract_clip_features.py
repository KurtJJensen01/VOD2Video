#!/usr/bin/env python3
"""CLI entry point for the Phase 2A clip feature pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.clip_features import (  # noqa: E402
    AudioExtractionConfig,
    DEFAULT_FEATURE_COLUMNS,
    ClipFeatureExtractionError,
    ClipSamplingConfig,
    build_feature_manifest,
    build_feature_manifest_summary,
    load_feature_source_manifest,
    resolve_audio_tool_status,
    write_feature_manifest_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract real clip-derived features from a split manifest."
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=REPO_ROOT / "artifacts" / "splits" / "branch_1c" / "all_splits.csv",
        help="Path to the combined split manifest CSV from Phase 1C.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "features" / "branch_2a",
        help="Directory where the feature CSV and summary JSON will be written.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Target low-rate frame sampling frequency used for visual features.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Maximum number of sampled frames per clip.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=160,
        help="Width used before computing grayscale summary features.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=90,
        help="Height used before computing grayscale summary features.",
    )
    parser.add_argument(
        "--keep-columns",
        nargs="*",
        default=["split_block_id", "start_time_seconds", "end_time_seconds"],
        help="Optional source-manifest columns to carry into the output CSV.",
    )
    parser.add_argument(
        "--show-head",
        type=int,
        default=5,
        help="How many extracted rows to print as a preview.",
    )
    parser.add_argument(
        "--disable-audio",
        action="store_true",
        help="Skip ffmpeg-based audio extraction and emit fallback audio columns.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for ffmpeg audio decode.",
    )
    parser.add_argument(
        "--audio-window-size",
        type=int,
        default=2048,
        help="Window size in samples for RMS/energy summaries.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.02,
        help="Absolute amplitude threshold used for the audio silence ratio.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        type=str,
        help="Optional explicit path to ffmpeg if it is not on PATH.",
    )
    parser.add_argument(
        "--ffprobe-path",
        type=str,
        help="Optional explicit path to ffprobe if it is not on PATH.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sampling_config = ClipSamplingConfig(
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
    )
    audio_config = AudioExtractionConfig(
        enabled=not args.disable_audio,
        target_sample_rate=args.audio_sample_rate,
        window_size_samples=args.audio_window_size,
        silence_threshold=args.silence_threshold,
        ffmpeg_path=args.ffmpeg_path,
        ffprobe_path=args.ffprobe_path,
    )
    audio_tool_status = resolve_audio_tool_status(audio_config)

    try:
        source_manifest = load_feature_source_manifest(args.split_manifest)
        feature_manifest = build_feature_manifest(
            source_manifest,
            sampling_config=sampling_config,
            audio_config=audio_config,
            include_optional_columns=args.keep_columns,
        )
        summary = build_feature_manifest_summary(
            feature_manifest,
            sampling_config=sampling_config,
            audio_config=audio_config,
            audio_tool_status=audio_tool_status,
        )
        output_paths = write_feature_manifest_outputs(
            feature_manifest,
            output_dir=args.output_dir,
            summary=summary,
        )
    except (ClipFeatureExtractionError, FileNotFoundError, ValueError) as exc:
        print(f"Feature extraction failed: {exc}", file=sys.stderr)
        return 1

    print("Feature extraction complete")
    print(f"source manifest={args.split_manifest.resolve()}")
    print(f"feature csv={output_paths['features_csv']}")
    print(f"summary json={output_paths['summary_json']}")
    print("feature columns=" + ", ".join(summary.feature_columns))
    print("split counts=" + json.dumps(summary.split_counts))
    print("label counts=" + json.dumps(summary.label_counts))
    print("sampling=" + json.dumps(summary.sampling))
    print("audio=" + json.dumps(summary.audio))
    if audio_tool_status.warning:
        print("warning=" + audio_tool_status.warning)
    print()
    preview_columns = [
        "unique_id",
        "split",
        "label",
        *DEFAULT_FEATURE_COLUMNS[:6],
        "audio_available",
        "audio_duration_seconds",
        "audio_peak_amplitude",
        "audio_silence_ratio",
        *DEFAULT_FEATURE_COLUMNS[-4:],
    ]
    available_columns = [column for column in preview_columns if column in feature_manifest.columns]
    if args.show_head > 0:
        print(feature_manifest[available_columns].head(args.show_head).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
