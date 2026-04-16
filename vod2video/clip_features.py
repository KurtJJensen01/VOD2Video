"""Clip-content feature extraction for Phase 2A.

This module builds a training-ready feature manifest from the Phase 1C split
output by reading the actual clip files and summarizing both visual and audio
signals from the clips themselves.

Visual features are extracted with OpenCV from low-rate sampled frames.
Audio features are extracted with ffmpeg by decoding the first audio stream to
mono PCM samples, then computing lightweight summary statistics. If ffmpeg is
not available, the pipeline continues in visual-only mode and emits documented
audio fallback values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

DEFAULT_ID_COLUMNS = (
    "unique_id",
    "vod_id",
    "segment_id",
    "label",
    "split",
    "clip_path",
    "resolved_clip_path",
)

DEFAULT_FEATURE_COLUMNS = (
    "video_duration_seconds",
    "video_fps",
    "video_frame_count",
    "video_width",
    "video_height",
    "sampled_frame_count",
    "sampled_frame_ratio",
    "brightness_mean",
    "brightness_std",
    "brightness_min",
    "brightness_max",
    "contrast_mean",
    "contrast_std",
    "motion_mean",
    "motion_std",
    "motion_max",
    "motion_p90",
    "audio_available",
    "audio_duration_seconds",
    "audio_sample_rate",
    "audio_rms_mean",
    "audio_rms_std",
    "audio_peak_amplitude",
    "audio_mean_amplitude",
    "audio_std_amplitude",
    "audio_silence_ratio",
    "audio_energy_mean",
    "audio_energy_std",
    "audio_energy_max",
    "audio_energy_p90",
)


class ClipFeatureExtractionError(RuntimeError):
    """Raised when a clip cannot be decoded into feature values."""


@dataclass(frozen=True)
class ClipSamplingConfig:
    """How video frames are sampled before feature computation."""

    sample_fps: float = 2.0
    max_frames: int = 16
    resize_width: int = 160
    resize_height: int = 90


@dataclass(frozen=True)
class AudioExtractionConfig:
    """How audio is decoded and summarized before feature computation."""

    enabled: bool = True
    target_sample_rate: int = 16000
    window_size_samples: int = 2048
    silence_threshold: float = 0.02
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


@dataclass(frozen=True)
class AudioToolStatus:
    """Resolved audio tool paths and whether audio extraction is available."""

    ffmpeg_path: str | None
    ffprobe_path: str | None
    audio_enabled: bool
    warning: str | None = None


@dataclass(frozen=True)
class FeatureManifestSummary:
    total_rows: int
    extracted_rows: int
    feature_columns: tuple[str, ...]
    split_counts: dict[str, int]
    label_counts: dict[int, int]
    sampling: dict[str, float | int]
    audio: dict[str, object]


def load_feature_source_manifest(path: Path) -> pd.DataFrame:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Feature source manifest not found: {manifest_path}")

    dataframe = pd.read_csv(manifest_path)
    missing = [column for column in DEFAULT_ID_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ClipFeatureExtractionError(
            f"Manifest is missing required column(s): {', '.join(missing)}"
        )

    return dataframe


def _safe_float(value: float | int | None, *, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return numeric


def _resolve_tool_path(configured_path: str | None, tool_name: str) -> str | None:
    if configured_path:
        candidate = Path(configured_path).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        return None
    return shutil.which(tool_name)


def resolve_audio_tool_status(
    config: AudioExtractionConfig | None = None,
) -> AudioToolStatus:
    active_config = config or AudioExtractionConfig()
    if not active_config.enabled:
        return AudioToolStatus(
            ffmpeg_path=None,
            ffprobe_path=None,
            audio_enabled=False,
            warning="Audio extraction disabled by configuration.",
        )

    ffmpeg_path = _resolve_tool_path(active_config.ffmpeg_path, "ffmpeg")
    ffprobe_path = _resolve_tool_path(active_config.ffprobe_path, "ffprobe")
    if ffmpeg_path is None:
        return AudioToolStatus(
            ffmpeg_path=None,
            ffprobe_path=ffprobe_path,
            audio_enabled=False,
            warning=(
                "ffmpeg was not found on PATH and no explicit --ffmpeg-path was provided. "
                "Continuing with visual-only features and audio fallback columns."
            ),
        )

    if ffprobe_path is None:
        return AudioToolStatus(
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=None,
            audio_enabled=True,
            warning=(
                "ffprobe was not found. Audio decoding will still run via ffmpeg, "
                "but preflight stream checks are unavailable."
            ),
        )

    return AudioToolStatus(
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        audio_enabled=True,
        warning=None,
    )


def make_audio_fallback_features(*, audio_available: bool = False) -> dict[str, float]:
    return {
        "audio_available": float(1.0 if audio_available else 0.0),
        "audio_duration_seconds": 0.0,
        "audio_sample_rate": 0.0,
        "audio_rms_mean": 0.0,
        "audio_rms_std": 0.0,
        "audio_peak_amplitude": 0.0,
        "audio_mean_amplitude": 0.0,
        "audio_std_amplitude": 0.0,
        "audio_silence_ratio": 1.0,
        "audio_energy_mean": 0.0,
        "audio_energy_std": 0.0,
        "audio_energy_max": 0.0,
        "audio_energy_p90": 0.0,
    }


def _probe_has_audio_stream(clip_path: Path, tool_status: AudioToolStatus) -> bool | None:
    if tool_status.ffprobe_path is None:
        return None

    command = [
        tool_status.ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(clip_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None

    streams = payload.get("streams", [])
    return bool(streams)


def _decode_audio_samples(
    clip_path: Path,
    config: AudioExtractionConfig,
    tool_status: AudioToolStatus,
) -> tuple[np.ndarray, int]:
    if tool_status.ffmpeg_path is None:
        raise ClipFeatureExtractionError("ffmpeg is required for audio decoding.")

    command = [
        tool_status.ffmpeg_path,
        "-v",
        "error",
        "-i",
        str(clip_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(config.target_sample_rate),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
        raise ClipFeatureExtractionError(
            f"ffmpeg audio decode failed for {clip_path.name}: {stderr_text or 'unknown error'}"
        )

    sample_buffer = result.stdout
    if not sample_buffer:
        return np.array([], dtype=np.float32), config.target_sample_rate

    samples = np.frombuffer(sample_buffer, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return np.array([], dtype=np.float32), config.target_sample_rate

    samples /= 32768.0
    return samples, config.target_sample_rate


def _window_audio(samples: np.ndarray, window_size: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((0, window_size), dtype=np.float32)

    usable_samples = samples.size - (samples.size % window_size)
    if usable_samples == 0:
        return samples.reshape(1, -1)

    trimmed = samples[:usable_samples]
    return trimmed.reshape(-1, window_size)


def extract_audio_features(
    clip_path: Path,
    *,
    audio_config: AudioExtractionConfig | None = None,
    tool_status: AudioToolStatus | None = None,
) -> dict[str, float]:
    """Extract simple audio summary features from a clip on disk."""

    active_config = audio_config or AudioExtractionConfig()
    active_status = tool_status or resolve_audio_tool_status(active_config)
    if not active_status.audio_enabled:
        return make_audio_fallback_features(audio_available=False)

    has_audio_stream = _probe_has_audio_stream(clip_path, active_status)
    if has_audio_stream is False:
        return make_audio_fallback_features(audio_available=False)

    try:
        samples, sample_rate = _decode_audio_samples(clip_path, active_config, active_status)
    except ClipFeatureExtractionError:
        return make_audio_fallback_features(audio_available=False)
    if samples.size == 0:
        return make_audio_fallback_features(audio_available=False)

    absolute_amplitude = np.abs(samples)
    windows = _window_audio(samples, max(1, active_config.window_size_samples))
    rms_windows = np.sqrt(np.mean(np.square(windows), axis=1)) if len(windows) else np.zeros(1, dtype=np.float32)
    energy_windows = np.mean(np.square(windows), axis=1) if len(windows) else np.zeros(1, dtype=np.float32)
    silence_ratio = float(np.mean(absolute_amplitude < active_config.silence_threshold))

    return {
        "audio_available": 1.0,
        "audio_duration_seconds": float(samples.size / max(sample_rate, 1)),
        "audio_sample_rate": float(sample_rate),
        "audio_rms_mean": float(rms_windows.mean()),
        "audio_rms_std": float(rms_windows.std()),
        "audio_peak_amplitude": float(absolute_amplitude.max()),
        "audio_mean_amplitude": float(absolute_amplitude.mean()),
        "audio_std_amplitude": float(absolute_amplitude.std()),
        "audio_silence_ratio": silence_ratio,
        "audio_energy_mean": float(energy_windows.mean()),
        "audio_energy_std": float(energy_windows.std()),
        "audio_energy_max": float(energy_windows.max()),
        "audio_energy_p90": float(np.percentile(energy_windows, 90)),
    }


def _compute_sample_indices(
    frame_count: int,
    fps: float,
    config: ClipSamplingConfig,
) -> np.ndarray:
    if frame_count <= 0:
        return np.array([], dtype=np.int32)

    if fps > 0:
        estimated_samples = max(1, int(np.ceil((frame_count / fps) * config.sample_fps)))
    else:
        estimated_samples = min(frame_count, config.max_frames)

    sample_count = max(1, min(frame_count, config.max_frames, estimated_samples))
    indices = np.linspace(0, frame_count - 1, num=sample_count, dtype=np.int32)
    return np.unique(indices)


def _read_sampled_grayscale_frames(
    clip_path: Path,
    config: ClipSamplingConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise ClipFeatureExtractionError(f"Could not open clip: {clip_path}")

    try:
        fps = _safe_float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(_safe_float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        width = int(round(_safe_float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(_safe_float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        sample_indices = set(_compute_sample_indices(frame_count, fps, config).tolist())
        if not sample_indices:
            raise ClipFeatureExtractionError(f"Clip has no readable frames: {clip_path}")

        sampled_frames: list[np.ndarray] = []
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in sample_indices:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(
                    gray_frame,
                    (config.resize_width, config.resize_height),
                    interpolation=cv2.INTER_AREA,
                )
                sampled_frames.append(resized.astype(np.float32) / 255.0)
                if len(sampled_frames) == len(sample_indices):
                    break
            frame_index += 1

        if not sampled_frames:
            raise ClipFeatureExtractionError(f"Failed to decode sampled frames from: {clip_path}")

        frames = np.stack(sampled_frames, axis=0)
        duration_seconds = (frame_count / fps) if fps > 0 else 0.0
        metadata = {
            "video_duration_seconds": float(duration_seconds),
            "video_fps": float(fps),
            "video_frame_count": float(frame_count),
            "video_width": float(width),
            "video_height": float(height),
            "sampled_frame_count": float(frames.shape[0]),
            "sampled_frame_ratio": float(frames.shape[0] / max(frame_count, 1)),
        }
        return frames, metadata
    finally:
        capture.release()


def extract_visual_features(
    clip_path: Path,
    *,
    sampling_config: ClipSamplingConfig | None = None,
) -> dict[str, float]:
    """Extract simple visual/activity features from a clip on disk."""

    config = sampling_config or ClipSamplingConfig()
    frames, metadata = _read_sampled_grayscale_frames(clip_path, config)

    brightness_per_frame = frames.mean(axis=(1, 2))
    contrast_per_frame = frames.std(axis=(1, 2))

    if frames.shape[0] >= 2:
        frame_deltas = np.abs(np.diff(frames, axis=0))
        motion_scores = frame_deltas.mean(axis=(1, 2))
    else:
        motion_scores = np.zeros(1, dtype=np.float32)

    return {
        **metadata,
        "brightness_mean": float(brightness_per_frame.mean()),
        "brightness_std": float(brightness_per_frame.std()),
        "brightness_min": float(brightness_per_frame.min()),
        "brightness_max": float(brightness_per_frame.max()),
        "contrast_mean": float(contrast_per_frame.mean()),
        "contrast_std": float(contrast_per_frame.std()),
        "motion_mean": float(motion_scores.mean()),
        "motion_std": float(motion_scores.std()),
        "motion_max": float(motion_scores.max()),
        "motion_p90": float(np.percentile(motion_scores, 90)),
    }


def extract_clip_features(
    row: pd.Series,
    *,
    sampling_config: ClipSamplingConfig | None = None,
    audio_config: AudioExtractionConfig | None = None,
    audio_tool_status: AudioToolStatus | None = None,
) -> dict[str, object]:
    """Extract a feature row while preserving the manifest identity columns."""

    clip_path = Path(str(row["resolved_clip_path"])).expanduser().resolve()
    if not clip_path.exists():
        raise FileNotFoundError(f"Resolved clip path not found: {clip_path}")

    base = {column: row[column] for column in row.index if column in DEFAULT_ID_COLUMNS}
    visual_features = extract_visual_features(clip_path, sampling_config=sampling_config)
    audio_features = extract_audio_features(
        clip_path,
        audio_config=audio_config,
        tool_status=audio_tool_status,
    )
    return {**base, **visual_features, **audio_features}


def build_feature_manifest(
    dataframe: pd.DataFrame,
    *,
    sampling_config: ClipSamplingConfig | None = None,
    audio_config: AudioExtractionConfig | None = None,
    include_optional_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Build one training-ready feature dataframe from a split manifest."""

    config = sampling_config or ClipSamplingConfig()
    active_audio_config = audio_config or AudioExtractionConfig()
    audio_tool_status = resolve_audio_tool_status(active_audio_config)
    output_rows: list[dict[str, object]] = []
    optional_columns = tuple(include_optional_columns)

    for row in dataframe.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        feature_row = extract_clip_features(
            row_series,
            sampling_config=config,
            audio_config=active_audio_config,
            audio_tool_status=audio_tool_status,
        )
        for column in optional_columns:
            if column in row_series.index and column not in feature_row:
                feature_row[column] = row_series[column]
        output_rows.append(feature_row)

    if not output_rows:
        return pd.DataFrame(columns=[*DEFAULT_ID_COLUMNS, *DEFAULT_FEATURE_COLUMNS, *optional_columns])

    feature_df = pd.DataFrame(output_rows)
    numeric_columns = [column for column in DEFAULT_FEATURE_COLUMNS if column in feature_df.columns]
    for column in numeric_columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="raise").astype("float32")

    if "label" in feature_df.columns:
        feature_df["label"] = pd.to_numeric(feature_df["label"], errors="raise").astype(int)

    ordered_columns = [
        *[column for column in DEFAULT_ID_COLUMNS if column in feature_df.columns],
        *[column for column in optional_columns if column in feature_df.columns and column not in DEFAULT_ID_COLUMNS],
        *[column for column in DEFAULT_FEATURE_COLUMNS if column in feature_df.columns],
    ]
    return feature_df.loc[:, ordered_columns].copy()


def build_feature_manifest_summary(
    dataframe: pd.DataFrame,
    *,
    sampling_config: ClipSamplingConfig,
    audio_config: AudioExtractionConfig,
    audio_tool_status: AudioToolStatus,
) -> FeatureManifestSummary:
    audio_available_rows = (
        int(pd.to_numeric(dataframe["audio_available"], errors="coerce").sum())
        if "audio_available" in dataframe.columns
        else 0
    )
    split_counts = (
        dataframe["split"].value_counts(sort=False).astype(int).to_dict()
        if "split" in dataframe.columns
        else {}
    )
    label_counts = (
        dataframe["label"].value_counts(sort=False).astype(int).to_dict()
        if "label" in dataframe.columns
        else {}
    )
    feature_columns = tuple(
        column for column in dataframe.columns if column in DEFAULT_FEATURE_COLUMNS
    )
    return FeatureManifestSummary(
        total_rows=int(len(dataframe)),
        extracted_rows=int(len(dataframe)),
        feature_columns=feature_columns,
        split_counts={str(key): int(value) for key, value in split_counts.items()},
        label_counts={int(key): int(value) for key, value in label_counts.items()},
        sampling=asdict(sampling_config),
        audio={
            "enabled": bool(audio_config.enabled),
            "ffmpeg_path": audio_tool_status.ffmpeg_path,
            "ffprobe_path": audio_tool_status.ffprobe_path,
            "audio_tool_enabled": bool(audio_tool_status.audio_enabled),
            "warning": audio_tool_status.warning,
            "audio_available_rows": audio_available_rows,
            "audio_unavailable_rows": int(len(dataframe) - audio_available_rows),
            "target_sample_rate": int(audio_config.target_sample_rate),
            "window_size_samples": int(audio_config.window_size_samples),
            "silence_threshold": float(audio_config.silence_threshold),
        },
    )


def write_feature_manifest_outputs(
    feature_dataframe: pd.DataFrame,
    *,
    output_dir: Path,
    summary: FeatureManifestSummary,
) -> dict[str, Path]:
    output_path = output_dir.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "features_csv": output_path / "clip_features.csv",
        "summary_json": output_path / "feature_summary.json",
    }
    feature_dataframe.to_csv(paths["features_csv"], index=False)
    paths["summary_json"].write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )
    return paths
