# Phase 5A: Final Metrics Package

**Note**: All file paths in this document are relative to the repository root directory. For example, `artifacts/model_improvement/...` refers to `<repo_root>/artifacts/model_improvement/...`.

## Final Model Selection
- **Experiment**: lower_learning_rate
- **Reason**: Achieved the highest validation F1 (0.516) and solid test F1 (0.424) among all Phase 4B experiments.
- **Checkpoint Path**: artifacts/model_improvement/branch_4b/runs/lower_learning_rate/best_model.pt

## Final Feature Setup
- **Feature Count**: 35
- **Features Used**: All available (video duration, FPS, frame count, dimensions, sampled frames, brightness/contrast/motion stats, audio RMS/energy/silence stats, start/end times, segment/vod indices)
- **Reason**: No feature improvements were tested in Phase 4A; baseline features performed adequately.

## Final Model Configuration
- **Learning Rate**: 0.0005
- **Hidden Dimension**: 32
- **Dropout**: 0.1
- **Epochs**: 12 (best at epoch 11)
- **Class Weight**: Enabled (positive class weight: 5.31)
- **Device**: CPU

## Final Decision Threshold / Selection Logic
- **Threshold**: 0.5 (default sigmoid output > 0.5 for positive class)
- **Selection Logic**: Rank clips by predicted probability descending; select top N for highlights.

## Final Metrics
- **Test Accuracy**: 0.747
- **Test Precision**: 0.326
- **Test Recall**: 0.609
- **Test F1**: 0.424
- **True Positives**: 14
- **True Negatives**: 98
- **False Positives**: 29
- **False Negatives**: 9

## Confusion Matrix
```
[[98, 29],
 [9, 14]]
```

## Experiment Comparison Table
See: artifacts/model_improvement/branch_4b/model_experiment_table.csv

## Final Observations
- The lower learning rate experiment outperformed the baseline by ~5% in test F1, indicating stable optimization.
- Class weighting improved performance; disabling it led to poor results (F1=0.2).
- Model shows moderate recall but lower precision, suggesting some false positives in highlight detection.
- Ready for Phase 6 inference pipeline with this locked configuration.</content>
<parameter name="filePath">/Users/drew/VOD2Video/outputs/phase_5a_final_metrics_package.md