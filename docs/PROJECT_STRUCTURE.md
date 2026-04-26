# Project Structure

This repo is organized around the active CNN+LSTM+MLP highlight-detection pipeline and several retained experiment paths.

## Root Files

- `README.md`: high-level project overview and run commands.
- `datasets.json`: explicit list of labeled dataset CSVs and clip roots used by the split pipeline.
- `requirements.txt`: practical Python dependencies for the active scripts and retained experiments.
- `.gitignore`: prevents clip folders, video files, archives, and generated artifacts from being committed.

## Source Folders

- `vod2video/`: reusable package code.
  - `dataset_loader.py`: validates and combines labeled CSV sources.
  - `dataset_split.py`: builds leakage-resistant train/val/test split manifests.
  - `clip_features.py`: extracts visual/audio summary features and writes the manifest used by training/evaluation.
  - `models.py`: contains the retained MLP model and the active CNN+LSTM+audio/MLP classifier model.
  - `training.py`, `training_data.py`, `training_config.py`, `checkpointing.py`: shared training utilities.
  - `evaluation.py`, `inference.py`, `metrics.py`: evaluation and scoring helpers.
  - `prediction_review.py`, `visualization.py`, `demo_selection.py`: reporting and presentation helpers.
  - `feature_improvement.py`, `model_improvement.py`: retained experiment helpers for the feature-based MLP path.

- `tools/`: runnable scripts.
  - Active default workflow:
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
  - Experiment scripts:
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
  - `demo_selection/`, `final_demo_package/`, `phase_5a_final_metrics/`: final review/demo artifacts.

- `docs/`: planning, proposal, structure, and handoff documentation.

- `outputs/`: older generated report/demo text outputs. Treat as generated material.

## Active Pipeline

The default path for current project artifacts is:

```bash
python tools/test_dataset_split.py --write-dir artifacts/splits/branch_1c
python tools/extract_clip_features.py --split-manifest artifacts/splits/branch_1c/all_splits.csv
python tools/run_real_baseline_training.py --feature-manifest artifacts/features/branch_2a/clip_features.csv --output-dir artifacts/training/branch_3a_real_baseline
python tools/review_predictions.py --prediction-csv artifacts/training/branch_3a_real_baseline/test_predictions.csv --output-dir artifacts/review/branch_3b
python tools/generate_result_visualizations.py --training-dir artifacts/training/branch_3a_real_baseline --review-dir artifacts/review/branch_3b --output-dir artifacts/visualization/branch_3c --split test
python tools/select_demo_examples.py --prediction-csv artifacts/training/branch_3a_real_baseline/test_predictions.csv --output-dir artifacts/demo_selection/branch_4c
python tools/build_final_demo_package.py --source-dir artifacts/demo_selection/branch_4c --output-dir artifacts/final_demo_package/branch_5b
```

## Experimental Work

The active model code path is `cnn_lstm_audio`: a pretrained ResNet18 frame encoder, an LSTM temporal encoder, and an MLP classifier head that also receives audio summary features. The feature-subset and model-improvement scripts are retained for the earlier MLP feature-experiment path. Hyperparameter-search scripts support the CNN+LSTM+MLP path.

## Legacy Work

Older phase outputs and support scripts are retained instead of deleted. They are useful for traceability but are not the default path for final results.
