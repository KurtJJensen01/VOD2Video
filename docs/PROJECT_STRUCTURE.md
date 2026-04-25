# Project Structure

This repo is organized around one active stable pipeline and several retained experiment paths.

## Root Files

- `README.md`: high-level project overview and run commands.
- `datasets.json`: explicit list of labeled dataset CSVs and clip roots used by the split pipeline.
- `requirements.txt`: practical Python dependencies for the active scripts and retained experiments.
- `.gitignore`: prevents clip folders, video files, archives, and generated artifacts from being committed.

## Source Folders

- `vod2video/`: reusable package code.
  - `dataset_loader.py`: validates and combines labeled CSV sources.
  - `dataset_split.py`: builds leakage-resistant train/val/test split manifests.
  - `clip_features.py`: extracts visual/audio summary features for the active MLP pipeline.
  - `models.py`: contains the active MLP model and retained experimental CNN+LSTM+audio model.
  - `training.py`, `training_data.py`, `training_config.py`, `checkpointing.py`: shared training utilities.
  - `evaluation.py`, `inference.py`, `metrics.py`: evaluation and scoring helpers.
  - `prediction_review.py`, `visualization.py`, `demo_selection.py`: reporting and presentation helpers.
  - `feature_improvement.py`, `model_improvement.py`: experiment helpers, not the default pipeline.

- `tools/`: runnable scripts.
  - Active stable workflow:
    - `test_dataset_split.py`
    - `extract_clip_features.py`
    - `run_real_baseline_training.py`
    - `review_predictions.py`
    - `generate_result_visualizations.py`
  - Dataset/support utilities:
    - `build_labeling_dataset.py`
    - `test_dataset_loader.py`
    - `score_feature_manifest.py`
    - `select_demo_examples.py`
  - Experimental scripts retained in place to avoid import breakage:
    - `train_baseline_model.py`
    - `run_feature_subset_experiments.py`
    - `run_model_improvement_experiments.py`
    - `run_hyperparameter_search.py`
    - `visualize_cnn_run.py`

- `labeling_test/`: labeled source data.
  - `*_Labels.csv` files are small source manifests and may be committed.
  - `*_Clips/` folders and `.7z` archives contain local video data and should not be committed.

- `artifacts/`: generated outputs.
  - `splits/`: split manifests.
  - `features/`: extracted feature manifests.
  - `training/`: checkpoints, metrics, prediction CSVs.
  - `review/`: prediction review outputs.
  - `visualization/`: generated charts and tables.
  - `feature_improvement/`, `model_improvement/`, `hyperparameter_search*/`: experiment outputs.

- `docs/`: planning, proposal, structure, and handoff documentation.

- `outputs/`: older generated report/demo text outputs. Treat as generated material.

## Active Pipeline

The stable path for final-project results is:

```bash
python tools/test_dataset_split.py --write-dir artifacts/splits/branch_1c
python tools/extract_clip_features.py --split-manifest artifacts/splits/branch_1c/all_splits.csv
python tools/run_real_baseline_training.py --feature-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/training/branch_3a_real_baseline
python tools/review_predictions.py --prediction-csv artifacts/training/branch_3a_real_baseline/test_predictions.csv --output-dir artifacts/review/branch_3b
python tools/generate_result_visualizations.py --training-dir artifacts/training/branch_3a_real_baseline --review-dir artifacts/review/branch_3b --output-dir artifacts/visualization/branch_3c --split test
```

## Experimental Work

CNN-only, CNN+LSTM, CNN+LSTM+audio, feature-subset, model-improvement, and hyperparameter-search work should be treated as experimental. The scripts and reusable code remain in their current locations because moving them would require import rewiring and could break old experiment reproduction.

## Legacy Work

Older phase outputs and support scripts are retained instead of deleted. They are useful for traceability but are not the default path for final results.
