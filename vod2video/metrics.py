"""Metric helpers for binary highlight classification."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_binary_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss: float,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).to(dtype=torch.int64)
    label_ints = labels.to(dtype=torch.int64)

    true_positives = int(((predictions == 1) & (label_ints == 1)).sum().item())
    true_negatives = int(((predictions == 0) & (label_ints == 0)).sum().item())
    false_positives = int(((predictions == 1) & (label_ints == 0)).sum().item())
    false_negatives = int(((predictions == 0) & (label_ints == 1)).sum().item())
    sample_count = int(labels.numel())

    accuracy = _safe_divide(true_positives + true_negatives, sample_count)
    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return BinaryClassificationMetrics(
        loss=float(loss),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        sample_count=sample_count,
    )
