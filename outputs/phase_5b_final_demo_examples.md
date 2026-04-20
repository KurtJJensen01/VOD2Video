# Phase 5B: Final Demo Example Package

## Overview
Curated set of prediction examples from the final model (lower_learning_rate) for demonstration purposes. Examples are selected from the test split to showcase model strengths and weaknesses.

## Best True Positives (High-confidence highlights correctly identified)
These are clips labeled as highlights (1) that the model strongly predicted as highlights.

- 2_seg_00476 (prob: 0.977) - Strong positive example
- 2_seg_00751 (prob: 0.879) - Clear highlight detection
- 1_seg_00039 (prob: 0.826) - Reliable prediction
- 2_seg_02297 (prob: 0.826) - Consistent performance
- 2_seg_02341 (prob: 0.784) - Good example

## Meaningful False Positives (High-confidence non-highlights misclassified)
These are clips labeled as non-highlights (0) that the model incorrectly predicted as highlights with high confidence.

- 2_seg_01194 (prob: 0.921) - Interesting misclassification
- 2_seg_00345 (prob: 0.834) - Shows model overconfidence
- 2_seg_00758 (prob: 0.811) - Edge case
- 2_seg_00994 (prob: 0.801) - Potential feature issue
- 1_seg_00675 (prob: 0.780) - Representative false positive

## Meaningful False Negatives (Low-confidence highlights missed)
These are clips labeled as highlights (1) that the model predicted as non-highlights with low confidence.

- 1_seg_01240 (prob: 0.332) - Missed highlight
- 1_seg_03148 (prob: 0.335) - Underscored importance
- 2_seg_05051 (prob: 0.342) - Potential feature gap
- 1_seg_00682 (prob: 0.375) - Borderline miss
- 1_seg_00656 (prob: 0.382) - Shows limitation

## Borderline Examples (Ambiguous predictions around 0.5)
These are clips where the model was uncertain, showing the decision boundary.

- 2_seg_02537 (prob: 0.592, label: 0, pred: 1) - Close call
- 2_seg_02665 (prob: 0.582, label: 0, pred: 1) - Model leaning positive
- 2_seg_02287 (prob: 0.580, label: 0, pred: 1) - Uncertain non-highlight
- 2_seg_03442 (prob: 0.568, label: 0, pred: 1) - Near threshold
- 1_seg_02176 (prob: 0.566, label: 1, pred: 1) - Correct but close

## Usage Notes
- All examples are from the test split to avoid data leakage.
- Clip paths can be resolved using the unique_id and the original manifest.
- This package is ready for final demo presentation.