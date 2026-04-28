"""Final video assembly for Phase 8.

Phase 8 is intentionally narrow: it reads the selected clip package produced by
Phase 7 and concatenates those clips into one final highlight MP4. It does not
train models, score clips, apply thresholds, or change clip timestamps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import wave

import pandas as pd


REQUIRED_SELECTION_COLUMNS: tuple[str, ...] = (
    "vod_id",
    "start_time_seconds",
    "end_time_seconds",
    "packaged_clip_path",
)

ASSEMBLY_MANIFEST_COLUMNS: tuple[str, ...] = (
    "assembly_order",
    "selection_rank",
    "unique_id",
    "vod_id",
    "segment_id",
    "start_time_seconds",
    "end_time_seconds",
    "packaged_clip_path",
    "resolved_packaged_clip_path",
)


class FinalVideoAssemblyError(ValueError):
    """Raised when Phase 8 assembly inputs are invalid or unbuildable."""


@dataclass(frozen=True)
class FinalVideoAssemblySummary:
    branch: str
    selection_manifest: str
    selection_dir: str
    output_dir: str
    output_video: str
    output_name: str
    order: str
    ffmpeg_path: str
    requested_reencode: bool
    stream_copy_attempted: bool
    reencode_attempted: bool
    used_reencode: bool
    total_clips: int
    total_source_duration_seconds: float | None
    concat_list: str
    assembly_manifest: str
    include_teaser: bool
    teaser_summary: str | None
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _resolve_ffmpeg(ffmpeg_path: Path | None) -> str:
    if ffmpeg_path is not None:
        expanded = ffmpeg_path.expanduser()
        if expanded.exists():
            return str(expanded.resolve())
        discovered = shutil.which(str(ffmpeg_path))
        if discovered:
            return discovered
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg_path}")

    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered
    raise FileNotFoundError(
        "ffmpeg not found. Install ffmpeg, add it to PATH, or pass --ffmpeg-path."
    )


def _resolve_ffprobe(ffmpeg_executable: str) -> str | None:
    ffmpeg_path = Path(ffmpeg_executable)
    sibling_names = ["ffprobe.exe", "ffprobe"]
    for sibling_name in sibling_names:
        ffprobe_path = ffmpeg_path.with_name(sibling_name)
        if ffprobe_path.exists():
            return str(ffprobe_path)
    return shutil.which("ffprobe")


def _resolve_clip_path(packaged_clip_path: object, selection_dir: Path) -> Path:
    if not isinstance(packaged_clip_path, str) or not packaged_clip_path.strip():
        raise FinalVideoAssemblyError("packaged_clip_path is empty for a selected clip.")

    raw_path = Path(packaged_clip_path)
    candidate = raw_path if raw_path.is_absolute() else selection_dir / raw_path
    return candidate.expanduser().resolve()


def _sort_manifest(manifest: pd.DataFrame, order: str) -> pd.DataFrame:
    sorted_manifest = manifest.copy()
    if order == "chronological":
        start_times = pd.to_numeric(
            sorted_manifest["start_time_seconds"], errors="coerce"
        )
        if start_times.isna().any():
            raise FinalVideoAssemblyError(
                "start_time_seconds must be numeric for chronological order."
            )
        sorted_manifest["_sort_start_time_seconds"] = start_times
        sorted_manifest = sorted_manifest.sort_values(
            by=["vod_id", "_sort_start_time_seconds"],
            kind="mergesort",
        ).drop(columns=["_sort_start_time_seconds"])
    elif order == "selection_rank":
        if "selection_rank" not in sorted_manifest.columns:
            raise FinalVideoAssemblyError(
                "selection_rank order requested, but selection_rank is missing."
            )
        selection_ranks = pd.to_numeric(
            sorted_manifest["selection_rank"], errors="coerce"
        )
        if selection_ranks.isna().any():
            raise FinalVideoAssemblyError(
                "selection_rank must be numeric when --order selection_rank is used."
            )
        sorted_manifest["_sort_selection_rank"] = selection_ranks
        sorted_manifest = sorted_manifest.sort_values(
            by=["_sort_selection_rank"],
            kind="mergesort",
        ).drop(columns=["_sort_selection_rank"])
    else:
        raise FinalVideoAssemblyError(f"Unsupported assembly order: {order}")

    return sorted_manifest.reset_index(drop=True)


def _format_concat_path(path: Path) -> str:
    safe_path = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{safe_path}'"


def _write_concat_list(paths: list[Path], concat_list_path: Path) -> None:
    lines = [_format_concat_path(path) for path in paths]
    concat_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_ffmpeg(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _stderr_tail(result: subprocess.CompletedProcess[str], max_chars: int = 2000) -> str:
    stderr = result.stderr.strip()
    if len(stderr) <= max_chars:
        return stderr
    return stderr[-max_chars:]


def _probe_duration_seconds(
    clip_path: Path,
    ffmpeg_executable: str,
    fallback_duration: float | None,
) -> float:
    ffprobe_executable = _resolve_ffprobe(ffmpeg_executable)
    if ffprobe_executable:
        result = subprocess.run(
            [
                ffprobe_executable,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(clip_path),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            try:
                duration = float(result.stdout.strip())
                if duration > 0:
                    return duration
            except ValueError:
                pass
    if fallback_duration is not None and fallback_duration > 0:
        return fallback_duration
    raise FinalVideoAssemblyError(f"Could not determine clip duration: {clip_path}")


def _has_audio_stream(clip_path: Path, ffmpeg_executable: str) -> bool:
    ffprobe_executable = _resolve_ffprobe(ffmpeg_executable)
    if ffprobe_executable is None:
        return True
    result = subprocess.run(
        [
            ffprobe_executable,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(clip_path),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _total_source_duration_seconds(manifest: pd.DataFrame) -> float | None:
    starts = pd.to_numeric(manifest["start_time_seconds"], errors="coerce")
    ends = pd.to_numeric(manifest["end_time_seconds"], errors="coerce")
    if starts.isna().any() or ends.isna().any():
        return None
    return float((ends - starts).sum())


def _select_teaser_rows(
    manifest: pd.DataFrame,
    teaser_clip_count: int,
    teaser_order: str,
) -> pd.DataFrame:
    count = max(int(teaser_clip_count), 0)
    if count <= 0:
        raise FinalVideoAssemblyError("--teaser-clip-count must be greater than 0.")

    if teaser_order == "score" and "predicted_probability" in manifest.columns:
        scores = pd.to_numeric(manifest["predicted_probability"], errors="coerce")
        if not scores.isna().all():
            candidates = manifest.copy()
            candidates["_teaser_score"] = scores.fillna(float("-inf"))
            return (
                candidates.sort_values(
                    by=["_teaser_score"], ascending=False, kind="mergesort"
                )
                .drop(columns=["_teaser_score"])
                .head(count)
                .reset_index(drop=True)
            )

    if "selection_rank" not in manifest.columns:
        raise FinalVideoAssemblyError(
            "Teaser ordering needs predicted_probability or selection_rank."
        )
    ranks = pd.to_numeric(manifest["selection_rank"], errors="coerce")
    if ranks.isna().any():
        raise FinalVideoAssemblyError("selection_rank must be numeric for teaser order.")
    candidates = manifest.copy()
    candidates["_teaser_selection_rank"] = ranks
    return (
        candidates.sort_values(by=["_teaser_selection_rank"], kind="mergesort")
        .drop(columns=["_teaser_selection_rank"])
        .head(count)
        .reset_index(drop=True)
    )


def _middle_start_seconds(duration_seconds: float, snippet_seconds: float) -> float:
    return max((duration_seconds - snippet_seconds) / 2.0, 0.0)


def _extract_audio_wav(
    *,
    ffmpeg_executable: str,
    clip_path: Path,
    wav_path: Path,
) -> bool:
    result = _run_ffmpeg(
        [
            ffmpeg_executable,
            "-y",
            "-i",
            str(clip_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(wav_path),
        ]
    )
    return result.returncode == 0 and wav_path.exists()


def _find_loudest_start_seconds(
    *,
    wav_path: Path,
    clip_duration_seconds: float,
    snippet_seconds: float,
) -> float:
    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        if sample_width != 2 or sample_rate <= 0 or frame_count <= 0:
            raise FinalVideoAssemblyError("Unsupported extracted WAV format.")
        raw = wav_file.readframes(frame_count)

    samples = [
        int.from_bytes(raw[i : i + 2], byteorder="little", signed=True)
        for i in range(0, len(raw), 2)
    ]
    window_size = max(int(sample_rate * snippet_seconds), 1)
    if len(samples) <= window_size:
        return 0.0

    step = max(int(sample_rate * 0.10), 1)
    best_start = 0
    best_energy = 0
    for start in range(0, len(samples) - window_size + 1, step):
        window = samples[start : start + window_size]
        energy = sum(sample * sample for sample in window)
        if energy > best_energy:
            best_energy = energy
            best_start = start

    if best_energy <= 0:
        raise FinalVideoAssemblyError("Audio is silent.")
    return min(best_start / sample_rate, max(clip_duration_seconds - snippet_seconds, 0.0))


def _extract_standardized_clip(
    *,
    ffmpeg_executable: str,
    input_path: Path,
    output_path: Path,
    start_seconds: float | None,
    duration_seconds: float | None,
) -> subprocess.CompletedProcess[str]:
    command = [ffmpeg_executable, "-y"]
    if start_seconds is not None:
        command.extend(["-ss", f"{start_seconds:.3f}"])
    command.extend(["-i", str(input_path)])
    has_audio = _has_audio_stream(input_path, ffmpeg_executable)
    if not has_audio:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
            ]
        )
    if duration_seconds is not None:
        command.extend(["-t", f"{duration_seconds:.3f}"])
    command.extend(["-map", "0:v:0"])
    if has_audio:
        command.extend(["-map", "0:a:0?"])
    else:
        command.extend(["-map", "1:a:0"])
    command.extend(
        [
            "-vf",
            "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,fps=30",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-shortest",
            str(output_path),
        ]
    )
    return _run_ffmpeg(command)


def _generate_black_separator(
    *,
    ffmpeg_executable: str,
    output_path: Path,
    duration_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return _run_ffmpeg(
        [
            ffmpeg_executable,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=640x360:r=30",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t",
            f"{duration_seconds:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    )


def _concat_reencoded(
    *,
    ffmpeg_executable: str,
    concat_list_path: Path,
    output_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_ffmpeg(
        [
            ffmpeg_executable,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )


def _build_teaser_intro(
    *,
    manifest: pd.DataFrame,
    resolved_clip_paths: list[Path],
    ffmpeg_executable: str,
    output_dir: Path,
    main_video_path: Path,
    final_output_path: Path,
    teaser_clip_count: int,
    teaser_snippet_seconds: float,
    teaser_order: str,
    teaser_snippet_mode: str,
    teaser_transition_seconds: float,
) -> dict[str, object]:
    if teaser_snippet_seconds <= 0:
        raise FinalVideoAssemblyError("--teaser-snippet-seconds must be greater than 0.")
    if teaser_transition_seconds < 0:
        raise FinalVideoAssemblyError("--teaser-transition-seconds cannot be negative.")

    teaser_dir = output_dir / "teaser_clips"
    teaser_dir.mkdir(parents=True, exist_ok=True)
    manifest_with_paths = manifest.copy()
    manifest_with_paths["_resolved_clip_path"] = [str(path) for path in resolved_clip_paths]
    teaser_rows = _select_teaser_rows(
        manifest_with_paths,
        teaser_clip_count=teaser_clip_count,
        teaser_order=teaser_order,
    )

    teaser_outputs: list[Path] = []
    clip_summaries: list[dict[str, object]] = []
    for teaser_index, row in enumerate(teaser_rows.to_dict(orient="records"), start=1):
        clip_path = Path(str(row["_resolved_clip_path"]))
        fallback_duration = None
        try:
            fallback_duration = float(row["end_time_seconds"]) - float(row["start_time_seconds"])
        except (KeyError, TypeError, ValueError):
            fallback_duration = None
        clip_duration = _probe_duration_seconds(
            clip_path, ffmpeg_executable, fallback_duration=fallback_duration
        )
        snippet_duration = min(float(teaser_snippet_seconds), clip_duration)
        mode_used = teaser_snippet_mode
        note = ""
        snippet_start = _middle_start_seconds(clip_duration, snippet_duration)
        if teaser_snippet_mode == "loudest":
            wav_path = teaser_dir / f"teaser_{teaser_index:02d}_audio.wav"
            try:
                if not _extract_audio_wav(
                    ffmpeg_executable=ffmpeg_executable,
                    clip_path=clip_path,
                    wav_path=wav_path,
                ):
                    raise FinalVideoAssemblyError("audio extraction failed")
                snippet_start = _find_loudest_start_seconds(
                    wav_path=wav_path,
                    clip_duration_seconds=clip_duration,
                    snippet_seconds=snippet_duration,
                )
            except Exception as exc:
                mode_used = "middle"
                note = f"loudest detection failed; used middle ({exc})"
                snippet_start = _middle_start_seconds(clip_duration, snippet_duration)

        snippet_path = teaser_dir / f"teaser_{teaser_index:02d}.mp4"
        result = _extract_standardized_clip(
            ffmpeg_executable=ffmpeg_executable,
            input_path=clip_path,
            output_path=snippet_path,
            start_seconds=snippet_start,
            duration_seconds=snippet_duration,
        )
        if result.returncode != 0 or not snippet_path.exists():
            raise FinalVideoAssemblyError(
                "ffmpeg failed while extracting teaser snippet. "
                f"stderr tail: {_stderr_tail(result)}"
            )
        teaser_outputs.append(snippet_path)
        clip_summaries.append(
            {
                "teaser_order": teaser_index,
                "source_clip": str(clip_path),
                "unique_id": row.get("unique_id"),
                "selection_rank": row.get("selection_rank"),
                "predicted_probability": row.get("predicted_probability"),
                "snippet_start_seconds": snippet_start,
                "snippet_duration_seconds": snippet_duration,
                "snippet_mode_requested": teaser_snippet_mode,
                "snippet_mode_used": mode_used,
                "output_clip": str(snippet_path),
                "note": note,
            }
        )

    separator_path = output_dir / "teaser_separator.mp4"
    separator_result = _generate_black_separator(
        ffmpeg_executable=ffmpeg_executable,
        output_path=separator_path,
        duration_seconds=float(teaser_transition_seconds),
    )
    if separator_result.returncode != 0 or not separator_path.exists():
        raise FinalVideoAssemblyError(
            "ffmpeg failed while generating teaser separator. "
            f"stderr tail: {_stderr_tail(separator_result)}"
        )

    standardized_main_path = output_dir / "main_highlight_video_standardized.mp4"
    standardize_result = _extract_standardized_clip(
        ffmpeg_executable=ffmpeg_executable,
        input_path=main_video_path,
        output_path=standardized_main_path,
        start_seconds=None,
        duration_seconds=None,
    )
    if standardize_result.returncode != 0 or not standardized_main_path.exists():
        raise FinalVideoAssemblyError(
            "ffmpeg failed while standardizing the main highlight video. "
            f"stderr tail: {_stderr_tail(standardize_result)}"
        )

    teaser_concat_list_path = output_dir / "teaser_concat_list.txt"
    _write_concat_list(
        [*teaser_outputs, separator_path, standardized_main_path],
        teaser_concat_list_path,
    )
    final_result = _concat_reencoded(
        ffmpeg_executable=ffmpeg_executable,
        concat_list_path=teaser_concat_list_path,
        output_path=final_output_path,
    )
    if final_result.returncode != 0 or not final_output_path.exists():
        raise FinalVideoAssemblyError(
            "ffmpeg failed while combining teaser and main video. "
            f"stderr tail: {_stderr_tail(final_result)}"
        )

    teaser_summary_path = output_dir / "teaser_summary.json"
    teaser_summary: dict[str, object] = {
        "include_teaser": True,
        "teaser_clip_count_requested": int(teaser_clip_count),
        "teaser_clip_count_used": len(teaser_outputs),
        "teaser_snippet_seconds": float(teaser_snippet_seconds),
        "teaser_order": teaser_order,
        "teaser_snippet_mode": teaser_snippet_mode,
        "teaser_transition_seconds": float(teaser_transition_seconds),
        "teaser_clips_dir": str(teaser_dir),
        "teaser_concat_list": str(teaser_concat_list_path),
        "separator_clip": str(separator_path),
        "main_highlight_video": str(main_video_path),
        "standardized_main_highlight_video": str(standardized_main_path),
        "final_output_video": str(final_output_path),
        "clips": clip_summaries,
    }
    teaser_summary_path.write_text(
        json.dumps(teaser_summary, indent=2, default=str), encoding="utf-8"
    )
    teaser_summary["teaser_summary"] = str(teaser_summary_path)
    return teaser_summary


def assemble_final_video(
    *,
    selection_manifest: Path,
    selection_dir: Path,
    output_dir: Path,
    output_name: str,
    order: str,
    ffmpeg_path: Path | None = None,
    reencode: bool = False,
    include_teaser: bool = False,
    teaser_clip_count: int = 3,
    teaser_snippet_seconds: float = 1.0,
    teaser_order: str = "score",
    teaser_snippet_mode: str = "loudest",
    teaser_transition_seconds: float = 0.5,
) -> tuple[pd.DataFrame, FinalVideoAssemblySummary, dict[str, Path]]:
    """Assemble Phase 7 selected clips into one Phase 8 highlight video."""

    resolved_manifest = selection_manifest.expanduser().resolve()
    resolved_selection_dir = selection_dir.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()

    if not resolved_manifest.exists():
        raise FileNotFoundError(f"Phase 7 selected manifest not found: {resolved_manifest}")

    ffmpeg_executable = _resolve_ffmpeg(ffmpeg_path)
    manifest = pd.read_csv(resolved_manifest)
    missing_columns = [c for c in REQUIRED_SELECTION_COLUMNS if c not in manifest.columns]
    if missing_columns:
        raise FinalVideoAssemblyError(
            f"Phase 7 manifest missing required column(s): {', '.join(missing_columns)}"
        )
    if manifest.empty:
        raise FinalVideoAssemblyError("No selected clips found in the Phase 7 manifest.")

    sorted_manifest = _sort_manifest(manifest, order=order)
    resolved_clip_paths = [
        _resolve_clip_path(path_value, resolved_selection_dir)
        for path_value in sorted_manifest["packaged_clip_path"].tolist()
    ]
    missing_clip_paths = [path for path in resolved_clip_paths if not path.exists()]
    if missing_clip_paths:
        first_missing = missing_clip_paths[0]
        raise FileNotFoundError(
            f"Selected clip file could not be found: {first_missing}"
            + (
                ""
                if len(missing_clip_paths) == 1
                else f" (+{len(missing_clip_paths) - 1} more)"
            )
        )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_video = resolved_output_dir / output_name
    main_output_video = (
        resolved_output_dir / "main_highlight_video.mp4" if include_teaser else output_video
    )
    concat_list_path = resolved_output_dir / "concat_list.txt"
    assembly_manifest_path = resolved_output_dir / "assembly_manifest.csv"
    assembly_summary_path = resolved_output_dir / "assembly_summary.json"

    _write_concat_list(resolved_clip_paths, concat_list_path)

    assembly_manifest = sorted_manifest.copy()
    assembly_manifest.insert(0, "assembly_order", range(1, len(assembly_manifest) + 1))
    assembly_manifest["resolved_packaged_clip_path"] = [
        str(path) for path in resolved_clip_paths
    ]
    leading = [c for c in ASSEMBLY_MANIFEST_COLUMNS if c in assembly_manifest.columns]
    trailing = [c for c in assembly_manifest.columns if c not in leading]
    assembly_manifest = assembly_manifest.loc[:, [*leading, *trailing]]
    assembly_manifest.to_csv(assembly_manifest_path, index=False)

    copy_command = [
        ffmpeg_executable,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c",
        "copy",
        str(main_output_video),
    ]
    reencode_command = [
        ffmpeg_executable,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        str(main_output_video),
    ]

    notes: list[str] = []
    stream_copy_attempted = not reencode
    reencode_attempted = reencode
    used_reencode = reencode
    final_result: subprocess.CompletedProcess[str]

    if reencode:
        final_result = _run_ffmpeg(reencode_command)
    else:
        copy_result = _run_ffmpeg(copy_command)
        if copy_result.returncode == 0:
            final_result = copy_result
        else:
            notes.append("Stream copy failed; retried with reencode.")
            reencode_attempted = True
            used_reencode = True
            final_result = _run_ffmpeg(reencode_command)

    if final_result.returncode != 0:
        raise FinalVideoAssemblyError(
            "ffmpeg failed while assembling the final video. "
            f"stderr tail: {_stderr_tail(final_result)}"
        )
    if not main_output_video.exists():
        raise FinalVideoAssemblyError(
            f"ffmpeg completed but output video was not created: {main_output_video}"
        )

    teaser_summary: dict[str, object] | None = None
    if include_teaser:
        teaser_summary = _build_teaser_intro(
            manifest=assembly_manifest,
            resolved_clip_paths=resolved_clip_paths,
            ffmpeg_executable=ffmpeg_executable,
            output_dir=resolved_output_dir,
            main_video_path=main_output_video,
            final_output_path=output_video,
            teaser_clip_count=teaser_clip_count,
            teaser_snippet_seconds=teaser_snippet_seconds,
            teaser_order=teaser_order,
            teaser_snippet_mode=teaser_snippet_mode,
            teaser_transition_seconds=teaser_transition_seconds,
        )

    summary = FinalVideoAssemblySummary(
        branch="8_final_video_assembly",
        selection_manifest=str(resolved_manifest),
        selection_dir=str(resolved_selection_dir),
        output_dir=str(resolved_output_dir),
        output_video=str(output_video),
        output_name=output_name,
        order=order,
        ffmpeg_path=ffmpeg_executable,
        requested_reencode=bool(reencode),
        stream_copy_attempted=stream_copy_attempted,
        reencode_attempted=reencode_attempted,
        used_reencode=used_reencode,
        total_clips=int(len(assembly_manifest)),
        total_source_duration_seconds=_total_source_duration_seconds(assembly_manifest),
        concat_list=str(concat_list_path),
        assembly_manifest=str(assembly_manifest_path),
        include_teaser=bool(include_teaser),
        teaser_summary=(
            str(teaser_summary["teaser_summary"]) if teaser_summary is not None else None
        ),
        notes=notes,
    )
    assembly_summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )

    output_paths: dict[str, Path] = {
        "output_video": output_video,
        "assembly_manifest_csv": assembly_manifest_path,
        "assembly_summary_json": assembly_summary_path,
        "concat_list": concat_list_path,
    }
    if teaser_summary is not None:
        output_paths["teaser_summary_json"] = Path(str(teaser_summary["teaser_summary"]))
        output_paths["main_highlight_video"] = main_output_video
    return assembly_manifest, summary, output_paths
