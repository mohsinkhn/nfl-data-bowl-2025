"""Configuration defaults for Approach 3 – target receiver pursuit model."""

from dataclasses import dataclass, field
from typing import List, Optional

from src.configs.config_approach1_v0 import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
)


@dataclass
class TargetDataConfig(DataConfig):
    """Data configuration tuned for the target-only dataset."""

    data_dir: str = "data/processed/fold1"
    dataset_type: str = "target_receiver"
    target_only: bool = True
    target_min_decoder_len: int = 5
    use_ball_residual_targets: bool = False
    use_player_role: bool = False
    use_player_attributes: bool = False
    use_ball_landing: bool = True  # Deprecated
    align_heading: bool = True
    rotation_normalize: bool = True
    rotation_augmentation_prob: float = 0.5
    rotation_augmentation_degrees: float = 10.0
    vertical_flip_prob: float = 0.5
    feature_cols: List[str] = field(
        default_factory=lambda: [
            "x",
            "y",
            "s",
            "a",
            "dir",
            "o",
            "ball_land_x",
            "ball_land_y",
            "ball_angle",
        ]
    )


@dataclass
class TargetModelConfig(ModelConfig):
    """Model configuration for the target pursuit regressor."""

    encoder_type: str = "gru"
    prediction_mode: str = "target_direct"
    decoder_type: str = "none"
    teacher_forcing_ratio: float = 0.0
    hidden_dim: int = 256
    dropout: float = 0.2
    ball_context_dim: int = 8
    ball_context_start: int = -8
    use_input_projection: bool = True
    input_projection_dim: Optional[int] = 256
    input_projection_dropout: float = 0.2


@dataclass
class TargetTrainingConfig(TrainingConfig):
    """Training configuration overrides for Approach 3."""

    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_dir: str = "checkpoints/approach3_target"
    log_dir: str = "logs/approach3_target"
    monitor_metric: str = "val_rmse"
    monitor_mode: str = "min"


def get_default_config():
    """Return default (data, model, training) configs for Approach 3."""

    return TargetDataConfig(), TargetModelConfig(), TargetTrainingConfig()
