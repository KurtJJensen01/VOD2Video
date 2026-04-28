# VOD2Video

**VOD2Video** is an AI-powered VOD-to-YouTube editing project. It segments livestream recordings, scores each segment for highlight quality, ranks the best moments, and assembles selected clips into a condensed highlight video.

This repository contains the machine learning, dataset, evaluation, inference, and artifact-generation pipeline for the deep learning course project. The final model is a binary highlight detector trained on labeled 5-second clips.

## Active Default Pipeline: CNN+LSTM+MLP

The active default model is the **CNN+LSTM+MLP highlight classifier**:

- CNN frame encoder: pretrained ResNet18 encodes sampled clip frames.
- LSTM temporal encoder: models the frame sequence across the clip.
- MLP classifier head: combines the LSTM video embedding with audio summary features and predicts highlight vs. non-highlight.

In code, this model is named `cnn_lstm_audio` because it includes audio features before the final MLP classifier. Some script and artifact names still include older words like `baseline` or `real_baseline`; treat those as legacy names. The implementation now builds the CNN+LSTM+MLP path.

## Final Artifact and Dataset Download

Large generated artifacts, labeled clip folders, checkpoints, VODs, and final videos are not committed to GitHub because of file size. They are provided through the final delivery SharePoint folder:

https://waynestateprod-my.sharepoint.com/:f:/g/personal/hx0783_wayne_edu/IgD8hUDnkfpOSYyry_Rjyni4AeFRZ2ayrzvMPkq7HVxar2E?e=sAD1K2

The shared folder contains:

- `VOD2Video_Final_Delivery/` with final artifacts, checkpoint, manifests, visualizations, inference outputs, selected highlights, and final video outputs.
- The four labeled clip folders / datasets needed by the project.
- The final generated highlight video.

Downloading this folder is the easiest way to inspect the submitted results or run later phases without rebuilding everything.

## Install

Clone the GitHub repo first, then install Python dependencies from the repo root:

```bash
pip install -r requirements.txt
```

Install FFmpeg and FFprobe for audio extraction, VOD segmentation, inference, and video assembly. The examples below use Windows paths:

```text
C:\ffmpeg\bin\ffmpeg.exe
C:\ffmpeg\bin\ffprobe.exe
```

Use your local FFmpeg/FFprobe paths if they are installed somewhere else.

## Where Downloaded Files Go

After cloning the repo, copy the downloaded artifacts and datasets into the repo root. The expected structure is:

```text
VOD2Video/
  artifacts/
    hyperparameter_search_broad/
      e100_lr0.0003_bs8/
        best_model.pt
    visualization/
    inference/
    highlight_selection/
    final_video/
  labeling_test/
    1_Jynxzi_Labels.csv
    2_Burnt_Peanut_Labels.csv
    3_Jynxzi_Labels.csv
    4_Burnt_Peanut_Labels.csv
    [matching clip folders/files as included in the download]
  datasets.json
  tools/
  vod2video/
  README.md
```

If the download contains a folder named `VOD2Video_Final_Delivery`, open it and copy the inner `artifacts/` folder and clip/dataset folders into the repo root.

Do not accidentally create `VOD2Video/VOD2Video_Final_Delivery/artifacts/...` unless you update paths manually. The commands in this README assume `artifacts/` and `labeling_test/` are directly inside the repo root.

`datasets.json` expects these CSVs:

- `labeling_test/1_Jynxzi_Labels.csv`
- `labeling_test/2_Burnt_Peanut_Labels.csv`
- `labeling_test/3_Jynxzi_Labels.csv`
- `labeling_test/4_Burnt_Peanut_Labels.csv`

The matching clip folders/files should also be under `labeling_test/`, as included in the download.

## Fast Path: Use Downloaded Artifacts

Use this path to inspect the final results, rerun final video assembly, or run inference/selection/video assembly with the downloaded checkpoint.

1. Clone the repo.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install FFmpeg and FFprobe.
4. Download the final delivery folder from the SharePoint link above.
5. Copy the downloaded `artifacts/` folder into the repo root.
6. Copy the labeled clip folders/datasets into the paths expected by `datasets.json`.
7. Confirm the checkpoint exists:

```text
artifacts/hyperparameter_search_broad/e100_lr0.0003_bs8/best_model.pt
```

8. Confirm the final video exists:

```text
artifacts/final_video/phase_8_final_demo_vod_10min_v2/final_highlight_video.mp4
```

### Rerun Phase 8 From Downloaded Phase 7 Outputs

```bash
python tools/assemble_final_video.py \
  --selection-manifest artifacts/highlight_selection/phase_7_final_demo_vod_10min_v2/selected_highlights_manifest.csv \
  --selection-dir artifacts/highlight_selection/phase_7_final_demo_vod_10min_v2 \
  --output-dir artifacts/final_video/phase_8_rerun_from_downloaded_artifacts \
  --output-name final_highlight_video.mp4 \
  --order chronological \
  --ffmpeg-path C:\ffmpeg\bin\ffmpeg.exe \
  --include-teaser \
  --teaser-clip-count 5 \
  --teaser-snippet-seconds 1.5 \
  --teaser-order score \
  --teaser-snippet-mode loudest \
  --teaser-transition-seconds 0.5
```

### Rerun Phases 6, 7, and 8 With Downloaded Checkpoint

Use a local VOD file for `--input`.

Phase 6:

```bash
python tools/run_vod_inference_pipeline.py \
  --input "path/to/vod.mp4" \
  --vod-id final_demo_vod \
  --output-dir artifacts/inference/phase_6_final_demo_vod \
  --checkpoint artifacts/hyperparameter_search_broad/e100_lr0.0003_bs8/best_model.pt \
  --threshold 0.60 \
  --segment-length 5 \
  --top-k 150 \
  --min-time-distance-seconds 20 \
  --ffmpeg-path C:\ffmpeg\bin\ffmpeg.exe \
  --ffprobe-path C:\ffmpeg\bin\ffprobe.exe
```

Phase 7:

```bash
python tools/select_highlight_clips.py \
  --input artifacts/inference/phase_6_final_demo_vod/inference/scored_clips.csv \
  --output-dir artifacts/highlight_selection/phase_7_final_demo_vod_10min_v2 \
  --threshold 0.35 \
  --top-k 120 \
  --min-gap-seconds 0
```

Phase 8:

```bash
python tools/assemble_final_video.py \
  --selection-manifest artifacts/highlight_selection/phase_7_final_demo_vod_10min_v2/selected_highlights_manifest.csv \
  --selection-dir artifacts/highlight_selection/phase_7_final_demo_vod_10min_v2 \
  --output-dir artifacts/final_video/phase_8_final_demo_vod_10min_v2 \
  --output-name final_highlight_video.mp4 \
  --order chronological \
  --ffmpeg-path C:\ffmpeg\bin\ffmpeg.exe \
  --include-teaser \
  --teaser-clip-count 5 \
  --teaser-snippet-seconds 1.5 \
  --teaser-order score \
  --teaser-snippet-mode loudest \
  --teaser-transition-seconds 0.5
```

## Full Reproduction Path From Scratch

Use this path to regenerate splits, features, the final checkpoint, visuals, inference outputs, selected clips, and the final video. This is slower than using the downloaded artifacts.

Exact regenerated metrics may vary slightly depending on hardware, random seeds, dependency versions, and CUDA behavior. The downloaded final checkpoint is the source used for the report/demo.

1. Clone the repo.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install FFmpeg and FFprobe.
4. Download the final delivery folder from the SharePoint link above.
5. Place the labeled clip folders and CSVs where `datasets.json` expects them, directly under `labeling_test/`.
6. Generate splits:

```bash
python tools/test_dataset_split.py --write-dir artifacts/splits/branch_1c
```

7. Extract features:

```bash
python tools/extract_clip_features.py --split-manifest artifacts/splits/branch_1c/all_splits.csv --output-dir artifacts/features/phase_5a_final_848_166_167
```

8. Regenerate the final checkpoint with the selected hyperparameters:

```bash
python tools/train_baseline_model.py \
  --split-manifest artifacts/splits/branch_1c/all_splits.csv \
  --output-dir artifacts/hyperparameter_search_broad/e100_lr0.0003_bs8 \
  --epochs 100 \
  --learning-rate 0.0003 \
  --batch-size 8 \
  --threshold 0.60 \
  --weight-decay 0.0001 \
  --patience 15 \
  --monitor-metric f1
```

9. Generate Phase 5A visuals:

```bash
python tools/review_predictions.py --prediction-csv artifacts/hyperparameter_search_broad/e100_lr0.0003_bs8/test_predictions.csv --output-dir artifacts/review/phase_5a_final_e100_lr0.0003_bs8 --threshold 0.60
python tools/generate_result_visualizations.py --training-dir artifacts/hyperparameter_search_broad/e100_lr0.0003_bs8 --review-dir artifacts/review/phase_5a_final_e100_lr0.0003_bs8 --output-dir artifacts/visualization/phase_5a_final_e100_lr0.0003_bs8 --split test
```

10. Run Phase 6 on a VOD:

```bash
python tools/run_vod_inference_pipeline.py \
  --input "path/to/vod.mp4" \
  --vod-id final_demo_vod \
  --output-dir artifacts/inference/phase_6_final_demo_vod \
  --checkpoint artifacts/hyperparameter_search_broad/e100_lr0.0003_bs8/best_model.pt \
  --threshold 0.60 \
  --segment-length 5 \
  --top-k 150 \
  --min-time-distance-seconds 20 \
  --ffmpeg-path C:\ffmpeg\bin\ffmpeg.exe \
  --ffprobe-path C:\ffmpeg\bin\ffprobe.exe
```

11. Run Phase 7 highlight selection:

```bash
python tools/select_highlight_clips.py \
  --input artifacts/inference/phase_6_final_demo_vod/inference/scored_clips.csv \
  --output-dir artifacts/highlight_selection/phase_7_final_demo_vod_10min_v2 \
  --threshold 0.35 \
  --top-k 120 \
  --min-gap-seconds 0
```

12. Run Phase 8 final video assembly:

```bash
python tools/assemble_final_video.py \
  --selection-manifest artifacts/highlight_selection/phase_7_final_demo_vod_10min_v2/selected_highlights_manifest.csv \
  --selection-dir artifacts/highlight_selection/phase_7_final_demo_vod_10min_v2 \
  --output-dir artifacts/final_video/phase_8_final_demo_vod_10min_v2 \
  --output-name final_highlight_video.mp4 \
  --order chronological \
  --ffmpeg-path C:\ffmpeg\bin\ffmpeg.exe \
  --include-teaser \
  --teaser-clip-count 5 \
  --teaser-snippet-seconds 1.5 \
  --teaser-order score \
  --teaser-snippet-mode loudest \
  --teaser-transition-seconds 0.5
```

## Artifact Verification Checklist

After copying downloaded files, these paths should exist:

- `artifacts/hyperparameter_search_broad/e100_lr0.0003_bs8/best_model.pt`
- `artifacts/visualization/phase_5a_final_e100_lr0.0003_bs8/test_confusion_matrix.png`
- `artifacts/inference/phase_6_final_demo_vod/inference/scored_clips.csv`
- `artifacts/highlight_selection/phase_7_final_demo_vod_10min_v2/selected_highlights_manifest.csv`
- `artifacts/final_video/phase_8_final_demo_vod_10min_v2/final_highlight_video.mp4`

## Command Reference

### Dataset Commands

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
python tools/extract_clip_features.py --split-manifest artifacts/splits/branch_1c/all_splits.csv --output-dir artifacts/features/phase_5a_final_848_166_167
```

Creates the training feature manifest. The default CNN+LSTM+MLP path uses this manifest for labels, splits, resolved clip paths, and audio-related inputs.

### Training and Scoring Commands

```bash
python tools/train_baseline_model.py --split-manifest artifacts/splits/branch_1c/all_splits.csv --output-dir artifacts/hyperparameter_search_broad/e100_lr0.0003_bs8 --epochs 100 --learning-rate 0.0003 --batch-size 8 --threshold 0.60 --weight-decay 0.0001 --patience 15 --monitor-metric f1
```

Required command to regenerate the final selected CNN+LSTM+audio run without rerunning the full hyperparameter grid. This writes the final checkpoint to `artifacts/hyperparameter_search_broad/e100_lr0.0003_bs8/best_model.pt`.

```bash
python tools/score_feature_manifest.py --checkpoint artifacts/hyperparameter_search_broad/e100_lr0.0003_bs8/best_model.pt --feature-manifest artifacts/splits/branch_1c/all_splits.csv --output-dir artifacts/inference/branch_2c --threshold 0.60
```

Scores a manifest with a saved checkpoint and writes ranked prediction outputs. Useful when you already have a checkpoint and do not need to retrain.

### Phase 6, 7, and 8 Commands

Phase 6 runs inference on a new unlabeled VOD and writes segmented clips, features, scored clips, top highlights, and a pipeline summary.

Phase 7 reads Phase 6 `scored_clips.csv`, applies threshold/top-k/min-gap selection, and packages selected clips for Phase 8.

Phase 8 reads the selected clip package from Phase 7 and merges those clips into one condensed highlight MP4. With `--include-teaser`, the final video starts with short snippets from the best selected clips, then a separator, then the full highlight video.

Use the exact Phase 6/7/8 commands in the fast path or full reproduction path above for final delivery outputs.

## Git Hygiene

Generated outputs are intentionally provided through the SharePoint download folder. Do not commit:

- `artifacts/`
- `vods/`
- Generated clips/videos
- `.pt` checkpoints
- `.7z` archives
- Bulk generated manifests, charts, or output packages

Small source files such as `datasets.json`, labeled CSV schemas, code, docs, and requirements should remain version controlled.

## Project Docs

- `docs/PROJECT_STRUCTURE.md`: file and folder map.
- `docs/TEAM_SCHEDULE.md`: team execution plan.
- `docs/VOD2Video_Project_Proposal.md`: academic proposal.
- `docs/VOD2Video_Technical_Project_Spec.md`: technical handoff/spec.
- `docs/VOD2Video_features.md`: feature reference and ideas.
