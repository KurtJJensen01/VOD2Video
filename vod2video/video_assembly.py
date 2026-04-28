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


def _total_source_duration_seconds(manifest: pd.DataFrame) -> float | None:
    starts = pd.to_numeric(manifest["start_time_seconds"], errors="coerce")
    ends = pd.to_numeric(manifest["end_time_seconds"], errors="coerce")
    if starts.isna().any() or ends.isna().any():
        return None
    return float((ends - starts).sum())


def assemble_final_video(
    *,
    selection_manifest: Path,
    selection_dir: Path,
    output_dir: Path,
    output_name: str,
    order: str,
    ffmpeg_path: Path | None = None,
    reencode: bool = False,
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
        str(output_video),
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
        str(output_video),
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
    if not output_video.exists():
        raise FinalVideoAssemblyError(
            f"ffmpeg completed but output video was not created: {output_video}"
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
    return assembly_manifest, summary, output_paths
