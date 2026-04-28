#!/usr/bin/env python3
"""CLI entry point for Phase 8 final video assembly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vod2video.video_assembly import (  # noqa: E402
    FinalVideoAssemblyError,
    assemble_final_video,
)


DEFAULT_SELECTION_MANIFEST = (
    REPO_ROOT
    / "artifacts"
    / "highlight_selection"
    / "phase_7"
    / "selected_highlights_manifest.csv"
)
DEFAULT_SELECTION_DIR = REPO_ROOT / "artifacts" / "highlight_selection" / "phase_7"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "final_video" / "phase_8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 8 final video assembly. Reads Phase 7 selected clips and "
            "concatenates them into one final highlight MP4. Optionally adds "
            "a short teaser montage before the full video."
        )
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
        help="Phase 7 selected_highlights_manifest.csv.",
    )
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=DEFAULT_SELECTION_DIR,
        help="Phase 7 output directory used to resolve packaged_clip_path values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where Phase 8 outputs are written.",
    )
    parser.add_argument(
        "--output-name",
        default="final_highlight_video.mp4",
        help="Filename for the assembled final MP4.",
    )
    parser.add_argument(
        "--order",
        choices=("chronological", "selection_rank"),
        default="chronological",
        help="Clip ordering strategy for assembly.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        type=Path,
        default=None,
        help="Optional path to ffmpeg. If omitted, ffmpeg is resolved from PATH.",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Skip stream copy and reencode the concat output with libx264/aac.",
    )
    parser.add_argument(
        "--include-teaser",
        action="store_true",
        help="Add a short teaser montage before the full highlight video.",
    )
    parser.add_argument(
        "--teaser-clip-count",
        type=int,
        default=3,
        help="Number of selected clips to use for the teaser.",
    )
    parser.add_argument(
        "--teaser-snippet-seconds",
        type=float,
        default=1.0,
        help="Length of each teaser snippet in seconds.",
    )
    parser.add_argument(
        "--teaser-order",
        choices=("score", "selection_rank"),
        default="score",
        help="How to choose teaser clips from the selected manifest.",
    )
    parser.add_argument(
        "--teaser-snippet-mode",
        choices=("loudest", "middle"),
        default="loudest",
        help="How to choose the snippet window inside each teaser clip.",
    )
    parser.add_argument(
        "--teaser-transition-seconds",
        type=float,
        default=0.5,
        help="Duration of the black separator between teaser and main video.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        _, summary, output_paths = assemble_final_video(
            selection_manifest=args.selection_manifest,
            selection_dir=args.selection_dir,
            output_dir=args.output_dir,
            output_name=args.output_name,
            order=args.order,
            ffmpeg_path=args.ffmpeg_path,
            reencode=args.reencode,
            include_teaser=args.include_teaser,
            teaser_clip_count=args.teaser_clip_count,
            teaser_snippet_seconds=args.teaser_snippet_seconds,
            teaser_order=args.teaser_order,
            teaser_snippet_mode=args.teaser_snippet_mode,
            teaser_transition_seconds=args.teaser_transition_seconds,
        )
    except (FileNotFoundError, FinalVideoAssemblyError, ValueError) as exc:
        print(f"Phase 8 final video assembly failed: {exc}", file=sys.stderr)
        return 1

    print("Phase 8 final video assembly complete")
    print(f"output dir={summary.output_dir}")
    print(f"final video={output_paths['output_video']}")
    print(f"assembly manifest={output_paths['assembly_manifest_csv']}")
    print(f"summary json={output_paths['assembly_summary_json']}")
    print(f"concat list={output_paths['concat_list']}")
    print(
        "assembly counts="
        + json.dumps(
            {
                "clips": summary.total_clips,
                "used_reencode": summary.used_reencode,
                "include_teaser": summary.include_teaser,
                "source_duration_seconds": summary.total_source_duration_seconds,
            }
        )
    )
    if "teaser_summary_json" in output_paths:
        print(f"teaser summary={output_paths['teaser_summary_json']}")
        print(f"main highlight video={output_paths['main_highlight_video']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
