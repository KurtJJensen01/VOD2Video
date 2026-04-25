# VOD2Video

**VOD2Video** is an AI-powered VOD-to-YouTube editor that turns long livestream recordings into condensed, long-form YouTube highlight videos.

The full project vision is to:
- take a full livestream VOD as input
- break it into short segments
- predict which segments are highlights
- rank the best moments
- generate a short teaser intro
- assemble a final long-form recap video

This repository is the machine learning and dataset pipeline for the deep learning course project. The current final-project path is highlight detection on labeled 5-second clips, with predictions and visualizations used for the report and presentation.

## Active Stable Pipeline: MLP Feature Baseline

The active default workflow is the **feature-based MLP baseline**. This path performed better than the CNN+LSTM experiments and should be treated as the stable pipeline for final results.

Run these commands from the repo root:

```bash
python tools/test_dataset_split.py --write-dir artifacts/splits/branch_1c
python tools/extract_clip_features.py --split-manifest artifacts/splits/branch_1c/all_splits.csv
python tools/run_real_baseline_training.py --feature-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/training/branch_3a_real_baseline
python tools/review_predictions.py --prediction-csv artifacts/training/branch_3a_real_baseline/test_predictions.csv --output-dir artifacts/review/branch_3b
python tools/generate_result_visualizations.py --training-dir artifacts/training/branch_3a_real_baseline --review-dir artifacts/review/branch_3b --output-dir artifacts/visualization/branch_3c --split test
```

That workflow does the following:

1. Build or add labeled 5-second clip datasets.
2. Generate a combined train/validation/test split manifest from all labeled CSVs.
3. Extract visual/audio summary features from clip files.
4. Train the MLP baseline on the feature manifest.
5. Review test predictions and errors.
6. Generate tables/charts for the report and presentation.

## Dataset Setup

Dataset sources are listed in root-level `datasets.json`. Each source has:

- `source_name`
- `csv_path`
- `clip_root`

`tools/test_dataset_split.py` prefers `datasets.json` when it exists. It still supports automatic discovery of `*_Labels.csv` files under `labeling_test/`, and it still supports explicit repeated `--csv` / `--clip-root` arguments.

The current labeled source CSVs live in `labeling_test/`:

- `1_Jynxzi_Labels.csv`
- `2_Burnt_Peanut_Labels.csv`
- `3_Jynxzi_Labels.csv`
- `4_Burnt_Peanut_Labels.csv`

The matching clip folders are local data and should not be committed.

## Generated Artifacts

Generated outputs go under `artifacts/` and are ignored by git. Regenerate them with the active workflow commands above.

Important generated locations:

- `artifacts/splits/branch_1c/`: train/val/test split manifests
- `artifacts/features/branch_2a/`: extracted clip feature manifest
- `artifacts/training/branch_3a_real_baseline/`: MLP training outputs, checkpoint, metrics, predictions
- `artifacts/review/branch_3b/`: prediction review and error-analysis outputs
- `artifacts/visualization/branch_3c/`: report/presentation visualizations
- `artifacts/feature_improvement/`, `artifacts/model_improvement/`, `artifacts/hyperparameter_search*/`: experiment outputs

Do not commit video clips, `.7z` clip archives, checkpoints, generated manifests, generated charts, or bulk generated outputs.

## Experimental Work

The CNN-only, CNN+LSTM, CNN+LSTM+audio, feature-subset, model-improvement, and hyperparameter-search work is useful project evidence, but it is **not** the main default workflow.

Experimental entry points currently include:

```bash
python tools/train_baseline_model.py --split-manifest artifacts/splits/branch_1c/all_splits.csv
python tools/run_feature_subset_experiments.py --feature-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/feature_improvement/branch_4a
python tools/run_model_improvement_experiments.py --feature-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/model_improvement/branch_4b
python tools/run_hyperparameter_search.py --split-manifest artifacts/splits/branch_1c/all_splits.csv --output-dir artifacts/hyperparameter_search
python tools/visualize_cnn_run.py --input-dir artifacts/hyperparameter_search --output-dir artifacts/visualization/cnn_run
```

The reusable CNN+LSTM+audio model code remains in `vod2video/models.py` and the video/audio dataloader support remains in `vod2video/training_data.py` because moving those files would risk breaking imports. Treat them as experimental unless a teammate is specifically working on that path.

## Legacy And Support Scripts

Some scripts are retained for reproducibility or older branch workflows:

- `tools/test_dataset_loader.py`: loader smoke-test utility
- `tools/score_feature_manifest.py`: older inference/scoring path for a saved checkpoint
- `tools/select_demo_examples.py`: later-stage demo selection from prediction artifacts
- `outputs/`: older generated report/demo text outputs, now treated as generated material

These are not deleted because they may still be useful for reproducing earlier project phases.

## Install

Create an environment, install dependencies, and make sure `ffmpeg`/`ffprobe` are on PATH if you want audio features:

```bash
pip install -r requirements.txt
```

If `ffmpeg` is missing, feature extraction will continue with visual features and documented audio fallback columns. You can also pass explicit tool paths:

```bash
python tools/extract_clip_features.py --ffmpeg-path C:\path\to\ffmpeg.exe --ffprobe-path C:\path\to\ffprobe.exe
```

## Project Docs

- `docs/PROJECT_STRUCTURE.md`: file and folder map
- `docs/TEAM_SCHEDULE.md`: team execution plan
- `docs/VOD2Video_Project_Proposal.md`: academic proposal
- `docs/VOD2Video_Technical_Project_Spec.md`: technical handoff/spec
- `docs/VOD2Video_features.md`: feature reference and ideas
