# VOD2Video

**VOD2Video** is an AI-powered VOD-to-YouTube editing project. The long-term product goal is to turn livestream recordings into condensed YouTube highlight videos by segmenting a VOD, scoring each segment for highlight quality, ranking the best moments, and using those ranked clips for review/demo packages.

This repository contains the machine learning, dataset, evaluation, and artifact-generation pipeline for the deep learning course project. The current working task is binary highlight detection on labeled 5-second clips.

## Active Default Pipeline: CNN+LSTM+MLP

The active default model is the **CNN+LSTM+MLP highlight classifier**:

- CNN frame encoder: pretrained ResNet18 encodes sampled clip frames.
- LSTM temporal encoder: models the frame sequence across the clip.
- MLP classifier head: combines the LSTM video embedding with audio summary features and predicts highlight vs. non-highlight.

In code, this model is named `cnn_lstm_audio` because it includes audio features before the final MLP classifier. Some script and artifact names still include older words like `baseline` or `real_baseline`; treat those as legacy names. The implementation now builds the CNN+LSTM+MLP path.

## Install

Create a Python environment, install dependencies, and make sure local clip files exist under the paths referenced by `datasets.json`.

```bash
pip install -r requirements.txt
```

Install `ffmpeg` and `ffprobe` if you want full audio extraction. If they are missing, feature extraction can still continue with audio fallback columns, and the training dataloader will use zero audio features when audio decode is unavailable.

You can pass explicit tool paths when extracting features:

```bash
python tools/extract_clip_features.py --ffmpeg-path C:\path\to\ffmpeg.exe --ffprobe-path C:\path\to\ffprobe.exe
```

## Regenerate The Main Artifacts

Run commands from the repo root.

```bash
python tools/test_dataset_split.py --write-dir artifacts/splits/branch_1c
python tools/extract_clip_features.py --split-manifest artifacts/splits/branch_1c/all_splits.csv --output-dir artifacts/features/branch_2a
python tools/run_real_baseline_training.py --feature-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/training/branch_3a_real_baseline
python tools/review_predictions.py --prediction-csv artifacts/training/branch_3a_real_baseline/test_predictions.csv --output-dir artifacts/review/branch_3b
python tools/generate_result_visualizations.py --training-dir artifacts/training/branch_3a_real_baseline --review-dir artifacts/review/branch_3b --output-dir artifacts/visualization/branch_3c --split test
python tools/select_demo_examples.py --prediction-csv artifacts/training/branch_3a_real_baseline/test_predictions.csv --output-dir artifacts/demo_selection/branch_4c
python tools/build_final_demo_package.py --source-dir artifacts/demo_selection/branch_4c --output-dir artifacts/final_demo_package/branch_5b
```

This produces the core project artifacts:

- `artifacts/splits/branch_1c/`: train/val/test manifests and split summaries.
- `artifacts/features/branch_2a/`: clip feature manifest and feature summary.
- `artifacts/training/branch_3a_real_baseline/`: CNN+LSTM+MLP checkpoint, metrics, training history, and prediction CSVs.
- `artifacts/review/branch_3b/`: ranked predictions, true/false positive tables, and review summary.
- `artifacts/visualization/branch_3c/`: report/presentation charts and tables.
- `artifacts/demo_selection/branch_4c/`: selected examples for demos and review.
- `artifacts/final_demo_package/branch_5b/`: copied demo clips plus `final_demo_manifest.csv` and package summary.

Other generated folders may exist from previous runs:

- `artifacts/phase_5a_final_metrics/`: locked final metrics package from the current academic project state.
- `artifacts/final_demo_package/branch_5b_review_check/`: review-check demo package variant.
- `outputs/`: older generated report/demo text outputs.

## Optional Hyperparameter Search

Use this when you need the broader CNN+LSTM+MLP experiment grid. It is much slower than the main artifact path because it trains many combinations.

```bash
python tools/run_hyperparameter_search.py --feature-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/hyperparameter_search_broad
```

Visualize one completed run by pointing at a run folder that contains `metrics.json`, `result.json`, and prediction CSVs:

```bash
python tools/visualize_cnn_run.py --run-dir artifacts/hyperparameter_search/e50_lr0.0001_bs4 --output-dir artifacts/visualization/cnn_best
```

## Dataset Setup

Dataset sources are listed in root-level `datasets.json`. Each source has:

- `source_name`
- `csv_path`
- `clip_root`

The current labeled CSVs live in `labeling_test/`:

- `1_Jynxzi_Labels.csv`
- `2_Burnt_Peanut_Labels.csv`
- `3_Jynxzi_Labels.csv`
- `4_Burnt_Peanut_Labels.csv`

The matching clip folders are local data and should not be committed. `tools/test_dataset_split.py` uses `datasets.json` by default, falls back to auto-discovering `*_Labels.csv` under `labeling_test/`, and also supports repeated `--csv` / `--clip-root` arguments.

## Command Reference

### Dataset and feature commands

```bash
python tools/build_labeling_dataset.py --input path\to\vod.mp4 --output-dir labeling_test\new_source --segment-length 5
```

Exports fixed-length clips and a labels CSV for manual labeling. This matters when adding new source VODs.

```bash
python tools/test_dataset_loader.py
```

Validates labeled CSVs, clip paths, required columns, and combined dataset loading. This is the fastest sanity check before splitting.

```bash
python tools/test_dataset_split.py --write-dir artifacts/splits/branch_1c
```

Builds repeatable train/val/test splits while keeping nearby clips in the same split block to reduce leakage.

```bash
python tools/extract_clip_features.py --split-manifest artifacts/splits/branch_1c/all_splits.csv --output-dir artifacts/features/branch_2a
```

Creates the training feature manifest. The default CNN+LSTM+MLP path uses this manifest for labels, splits, resolved clip paths, and audio-related inputs.

### Training and scoring commands

```bash
python tools/run_real_baseline_training.py --feature-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/training/branch_3a_real_baseline
```

Trains the current CNN+LSTM+MLP implementation and writes checkpoints, metrics, and train/val/test predictions. The folder name is legacy.

```bash
python tools/train_baseline_model.py --split-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/training/cnn_lstm_audio
```

Lower-level training entry point for the same `cnn_lstm_audio` model. Useful for quick custom training runs.

```bash
python tools/score_feature_manifest.py --checkpoint artifacts/training/branch_3a_real_baseline/best_model.pt --feature-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/inference/branch_2c
```

Scores a manifest with a saved checkpoint and writes ranked prediction outputs. Useful when you already have a checkpoint and do not need to retrain.

### Review, visualization, and demo commands

```bash
python tools/review_predictions.py --prediction-csv artifacts/training/branch_3a_real_baseline/test_predictions.csv --output-dir artifacts/review/branch_3b
```

Ranks predictions and separates true positives, false positives, true negatives, and false negatives for model review.

```bash
python tools/generate_result_visualizations.py --training-dir artifacts/training/branch_3a_real_baseline --review-dir artifacts/review/branch_3b --output-dir artifacts/visualization/branch_3c --split test
```

Builds presentation-ready charts and tables from training and review artifacts.

```bash
python tools/select_demo_examples.py --prediction-csv artifacts/training/branch_3a_real_baseline/test_predictions.csv --output-dir artifacts/demo_selection/branch_4c
```

Selects high-signal examples for demos: top highlights, true positives, false positives, false negatives, and borderline cases.

```bash
python tools/build_final_demo_package.py --source-dir artifacts/demo_selection/branch_4c --output-dir artifacts/final_demo_package/branch_5b
```

Copies selected clips into a category-organized final demo package and writes a manifest/summary.

### Experiment commands

```bash
python tools/run_feature_subset_experiments.py --feature-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/feature_improvement/branch_4a
python tools/run_model_improvement_experiments.py --feature-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/model_improvement/branch_4b
python tools/run_hyperparameter_search.py --feature-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/hyperparameter_search_broad
python tools/visualize_cnn_run.py --run-dir artifacts/hyperparameter_search/e50_lr0.0001_bs4 --output-dir artifacts/visualization/cnn_best
```

These commands support experiment comparison and presentation evidence. The feature/model improvement scripts are retained for the MLP feature-experiment path, while the hyperparameter search and CNN visualization scripts support the CNN+LSTM+MLP path.

## Generated Artifacts And Git Hygiene

Generated outputs go under `artifacts/` and are ignored by git. Do not commit video clips, `.7z` clip archives, checkpoints, generated manifests, generated charts, or bulk generated outputs.
To run the Phase 6 end-to-end inference pipeline on a new unlabeled VOD and
produce ranked highlight candidates, run:

```bash
python tools/run_vod_inference_pipeline.py \
  --input path/to/vod.mp4 \
  --vod-id final_test \
  --checkpoint artifacts/hyperparameter_search_broad/e100_lr0.0003_bs8/best_model.pt \
  --threshold 0.60 \
  --top-k 20 \
  --min-time-distance-seconds 30
```

`--threshold` overrides the value saved in the checkpoint. Omit it to use the
checkpoint's default. `--min-time-distance-seconds` enforces a minimum gap
between any two predicted highlights in `top_highlights.csv` so the top picks
are spread across the VOD rather than clustering in one moment.

If `ffmpeg`/`ffprobe` are not on PATH, pass explicit paths:

```bash
python tools/run_vod_inference_pipeline.py \
  --input path/to/vod.mp4 \
  --vod-id 3 \
  --ffmpeg-path C:\path\to\ffmpeg.exe \
  --ffprobe-path C:\path\to\ffprobe.exe
```

Output is written to `artifacts/inference/phase_6/` by default:
- `clips/` — 5-second MP4 segments
- `clip_manifest.csv` — per-clip metadata
- `features/clip_features.csv` — extracted feature vectors
- `inference/scored_clips.csv` — all scored clips ranked by predicted probability
- `inference/top_highlights.csv` — up to `--top-k` clips above the active threshold, filtered by `--min-time-distance-seconds`
- `pipeline_summary.json` — run parameters and output paths

--- 

Small source files such as `datasets.json`, labeled CSVs, code, docs, and requirements should remain version controlled. Treat `outputs/` as generated material too.

## Project Docs

- `docs/PROJECT_STRUCTURE.md`: file and folder map.
- `docs/TEAM_SCHEDULE.md`: team execution plan.
- `docs/VOD2Video_Project_Proposal.md`: academic proposal.
- `docs/VOD2Video_Technical_Project_Spec.md`: technical handoff/spec.
- `docs/VOD2Video_features.md`: feature reference and ideas.
