#!/usr/bin/env python3
"""Export fixed-length VOD segments plus a labeling CSV, preserving audio."""
# python tools/getTrainingSet.py --input "VIDEO FILE PATH" --output-dir "./labeling_test" --segment-length 5 --limit 20 (LIMIT IS OPTIONAL)

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary not found in PATH: {name}")


def get_video_duration_seconds(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}:\n{result.stderr}")

    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {result.stdout!r}") from exc


def seconds_to_hhmmss(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_segments(duration_seconds: float, segment_length: int) -> Iterable[tuple[int, float, float]]:
    segment_count = math.ceil(duration_seconds / segment_length)
    for index in range(segment_count):
        start = index * segment_length
        end = min((index + 1) * segment_length, duration_seconds)
        yield index, float(start), float(end)


def export_segment(
    input_video: Path,
    output_clip: Path,
    start_seconds: float,
    clip_duration: float,
    width: int | None,
    height: int | None,
    fps: int | None,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(input_video),
        "-t",
        f"{clip_duration:.3f}",
    ]

    filters: list[str] = []
    if width and height:
        filters.append(f"scale={width}:{height}")
    if fps:
        filters.append(f"fps={fps}")
    if filters:
        command.extend(["-vf", ",".join(filters)])

    command.extend([
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_clip),
    ])

    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while exporting {output_clip.name}:\n{result.stderr}"
        )


def write_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "segment_id",
        "clip_name",
        "clip_path",
        "start_time_seconds",
        "end_time_seconds",
        "start_time_hhmmss",
        "end_time_hhmmss",
        "label",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-length VOD segments for manual labeling.")
    parser.add_argument("--input", required=True, help="Path to input VOD video file")
    parser.add_argument("--output-dir", required=True, help="Folder to write clips and labels.csv into")
    parser.add_argument("--segment-length", type=int, default=5, help="Segment length in seconds (default: 5)")
    parser.add_argument("--width", type=int, default=640, help="Output clip width for review clips (default: 640)")
    parser.add_argument("--height", type=int, default=360, help="Output clip height for review clips (default: 360)")
    parser.add_argument("--fps", type=int, default=12, help="Output FPS for review clips (default: 12)")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of segments to export")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    require_binary("ffmpeg")
    require_binary("ffprobe")

    input_video = Path(args.input).expanduser().resolve()
    if not input_video.exists():
        print(f"Input file not found: {input_video}", file=sys.stderr)
        return 1

    if args.segment_length <= 0:
        print("--segment-length must be greater than 0", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    clips_dir = output_dir / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    duration_seconds = get_video_duration_seconds(input_video)
    print(f"Video duration: {duration_seconds:.2f} seconds ({seconds_to_hhmmss(duration_seconds)})")

    rows: list[dict[str, str]] = []
    segments = list(build_segments(duration_seconds, args.segment_length))
    if args.limit is not None:
        segments = segments[: args.limit]

    total_segments = len(segments)
    print(f"Exporting {total_segments} segment(s) to: {clips_dir}")

    for index, start_seconds, end_seconds in segments:
        clip_duration = end_seconds - start_seconds
        segment_id = f"seg_{index + 1:05d}"
        clip_name = f"{segment_id}_{int(start_seconds):06d}_{int(end_seconds):06d}.mp4"
        output_clip = clips_dir / clip_name

        print(
            f"[{index + 1}/{total_segments}] Exporting {clip_name} "
            f"({seconds_to_hhmmss(start_seconds)} -> {seconds_to_hhmmss(end_seconds)})"
        )

        export_segment(
            input_video=input_video,
            output_clip=output_clip,
            start_seconds=start_seconds,
            clip_duration=clip_duration,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )

        rows.append(
            {
                "segment_id": segment_id,
                "clip_name": clip_name,
                "clip_path": str(output_clip.relative_to(output_dir)),
                "start_time_seconds": f"{start_seconds:.3f}",
                "end_time_seconds": f"{end_seconds:.3f}",
                "start_time_hhmmss": seconds_to_hhmmss(start_seconds),
                "end_time_hhmmss": seconds_to_hhmmss(end_seconds),
                "label": "",
                "notes": "",
            }
        )

    csv_path = output_dir / "labels.csv"
    write_csv(csv_path, rows)

    print("Done.")
    print(f"Clips folder: {clips_dir}")
    print(f"CSV file:     {csv_path}")
    print("Fill in the label column with 1 for highlight or 0 for non-highlight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
