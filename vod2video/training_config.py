"""Configuration objects for the Phase 2B training framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Configuration for building dataloaders from a split manifest."""

    split_manifest_path: Path
    split_column: str = "split"
    label_column: str = "label"
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    train_split_name: str = "train"
    val_split_name: str = "val"
    test_split_name: str = "test"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the current placeholder baseline model."""

    model_name: str = "mlp_baseline"
    input_dim: int = 4
    hidden_dim: int = 16
    dropout: float = 0.1


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for model checkpoint persistence."""

    output_dir: Path
    monitor_metric: str = "f1"
    monitor_mode: str = "max"
    save_latest: bool = True
    best_filename: str = "best_model.pt"
    latest_filename: str = "latest_model.pt"
    history_filename: str = "training_history.json"


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level training hyperparameters and runtime settings."""

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    random_seed: int = 42
    device: str = "cpu"
    decision_threshold: float = 0.5
    positive_class_weight: float | None = None
    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(output_dir=Path("artifacts") / "training" / "baseline")
    )

    def to_serializable_dict(self) -> dict[str, object]:
        data = asdict(self)
        checkpoint = data["checkpoint"]
        checkpoint["output_dir"] = str(self.checkpoint.output_dir)
        return data
