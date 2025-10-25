"""Configuration classes for Approach 1 Baseline."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    # Paths
    data_dir: str = "data/processed/fold1"
    dataset_type: str = "baseline"  # 'baseline' or 'target_receiver'

    # Sequence lengths
    max_encoder_len: int = 40  # Based on analysis: 90th percentile
    max_decoder_len: int = 32  # Based on analysis recommendation

    # Features
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
    use_ball_landing: bool = True  # Deprecated; retained for compatibility.

    # Normalization options
    normalize: bool = True
    rotation_normalize: bool = True
    field_dims: tuple = (120.0, 53.3)  # NFL field dimensions

    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Additional features
    use_player_role: bool = True  # Whether to use player role encoding
    use_player_attributes: bool = True  # Whether to include player weight/height
    use_polar_targets: bool = False  # Predict delta time/angle instead of x,y
    align_heading: bool = False  # Do not align heading by default
    rotation_augmentation_prob: float = 0.0  # Probability of random rotation augmentation (train only)
    rotation_augmentation_degrees: float = 45.0  # Max absolute degrees for rotation augmentation
    vertical_flip_prob: float = 0.0  # Probability of flipping across horizontal axis
    target_only: bool = False  # Filter to targeted receivers (Approach 3)
    target_min_decoder_len: int = 1  # Minimum decoder frames required
    use_ball_residual_targets: bool = False  # Train on residual offsets to landing spot

    def __post_init__(self):
        """Validate configuration."""
        if self.max_encoder_len <= 0:
            raise ValueError(
                f"max_encoder_len must be positive, got {self.max_encoder_len}"
            )
        if self.max_decoder_len <= 0:
            raise ValueError(
                f"max_decoder_len must be positive, got {self.max_decoder_len}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if not 0.0 <= self.rotation_augmentation_prob <= 1.0:
            raise ValueError(
                f"rotation_augmentation_prob must be in [0, 1], got {self.rotation_augmentation_prob}"
            )
        if self.rotation_augmentation_degrees < 0.0:
            raise ValueError(
                f"rotation_augmentation_degrees must be non-negative, got {self.rotation_augmentation_degrees}"
            )
        if self.dataset_type not in {"baseline", "target_receiver"}:
            raise ValueError(
                f"dataset_type must be 'baseline' or 'target_receiver', got {self.dataset_type}"
            )
        if not 0.0 <= self.vertical_flip_prob <= 1.0:
            raise ValueError(
                f"vertical_flip_prob must be in [0, 1], got {self.vertical_flip_prob}"
            )
        if self.target_min_decoder_len <= 0:
            raise ValueError(
                f"target_min_decoder_len must be positive, got {self.target_min_decoder_len}"
            )
        if not isinstance(self.use_ball_residual_targets, bool):
            raise ValueError("use_ball_residual_targets must be boolean")


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Input/Output dimensions
    input_dim: int = 12  # base features + derived landing/length features (angles as sin/cos)
    output_dim: int = 2  # x, y

    # Architecture
    encoder_type: str = "lstm"  # 'lstm' or 'gru'
    decoder_type: str = "lstm"  # 'lstm' or 'gru'
    prediction_mode: str = "autoregressive"  # 'autoregressive' or 'direct'
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False

    # Decoder options
    teacher_forcing_ratio: float = 0.5
    use_attention: bool = False  # Simple attention mechanism

    # Ball landing context
    use_ball_context: bool = False  # Concatenate ball landing to decoder input
    ball_context_dim: int = 0  # Number of context features captured from encoder inputs
    ball_context_start: int = -4  # Inclusive index used to slice context from encoder inputs
    use_input_projection: bool = True
    input_projection_dim: Optional[int] = None
    input_projection_dropout: float = 0.1
    input_projection_batchnorm: bool = True
    encoder_input_dim: int = 0

    def __post_init__(self):
        """Validate configuration."""
        if self.encoder_type not in ["lstm", "gru"]:
            raise ValueError(
                f"encoder_type must be 'lstm' or 'gru', got {self.encoder_type}"
            )
        if self.decoder_type not in ["lstm", "gru", "none"]:
            raise ValueError(
                f"decoder_type must be 'lstm', 'gru', or 'none', got {self.decoder_type}"
            )
        if self.prediction_mode not in ["autoregressive", "direct", "target_direct"]:
            raise ValueError(
                f"prediction_mode must be 'autoregressive', 'direct', or 'target_direct', got {self.prediction_mode}"
            )
        if self.prediction_mode == "autoregressive" and self.decoder_type == "none":
            raise ValueError(
                "decoder_type cannot be 'none' when prediction_mode is 'autoregressive'"
            )
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0 <= self.teacher_forcing_ratio <= 1:
            raise ValueError(
                f"teacher_forcing_ratio must be in [0, 1], got {self.teacher_forcing_ratio}"
            )
        if self.prediction_mode in {"direct", "target_direct"}:
            self.teacher_forcing_ratio = 0.0
        if self.prediction_mode == "target_direct":
            self.decoder_type = "none"
        if self.use_input_projection:
            if self.input_projection_dim is None:
                self.input_projection_dim = self.hidden_dim
            if self.input_projection_dim <= 0:
                raise ValueError("input_projection_dim must be positive when projection is enabled")
            self.encoder_input_dim = self.input_projection_dim
        else:
            self.encoder_input_dim = self.input_dim
        if self.ball_context_dim < 0:
            raise ValueError(
                f"ball_context_dim must be non-negative, got {self.ball_context_dim}"
            )


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    # Training duration
    max_epochs: int = 50
    early_stopping_patience: int = 10

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_val: float = 1.0

    # Scheduler
    scheduler: str = "reduce_on_plateau"  # 'reduce_on_plateau', 'cosine', 'cosine_warmup', or 'none'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    scheduler_warmup_epochs: int = 5

    # Checkpointing
    checkpoint_dir: str = "checkpoints/approach1_v0"
    save_top_k: int = 3
    monitor_metric: str = "val_rmse"
    monitor_mode: str = "min"

    # Logging
    log_dir: str = "logs/approach1_v0"
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0  # Check validation every N epochs

    # Hardware
    accelerator: str = "auto"  # 'auto', 'gpu', 'cpu'
    devices: int = 1
    precision: str = "32"  # '32', '16', or 'bf16'

    # Reproducibility
    seed: Optional[int] = 42

    def __post_init__(self):
        """Validate configuration."""
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.scheduler not in ["reduce_on_plateau", "cosine", "none"]:
            raise ValueError(
                f"scheduler must be 'reduce_on_plateau', 'cosine', 'cosine_warmup', or 'none', got {self.scheduler}"
            )
        if self.monitor_mode not in ["min", "max"]:
            raise ValueError(
                f"monitor_mode must be 'min' or 'max', got {self.monitor_mode}"
            )
        if self.scheduler_warmup_epochs < 0:
            raise ValueError(
                f"scheduler_warmup_epochs must be >= 0, got {self.scheduler_warmup_epochs}"
            )


def get_default_config():
    """Get default configuration objects for Approach 1 baseline.

    Returns:
        Tuple of (DataConfig, ModelConfig, TrainingConfig) with default values.
    """
    return DataConfig(), ModelConfig(), TrainingConfig()


def get_fast_debug_config():
    """Get configuration for fast debugging/testing.

    Returns smaller model, fewer epochs, and smaller batch for quick iteration.

    Returns:
        Tuple of (DataConfig, ModelConfig, TrainingConfig) for debugging.
    """
    data_config = DataConfig(
        max_encoder_len=20,
        max_decoder_len=16,
        batch_size=8,
        num_workers=0,
    )

    model_config = ModelConfig(
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
    )

    training_config = TrainingConfig(
        max_epochs=5,
        early_stopping_patience=3,
        log_every_n_steps=10,
        checkpoint_dir="checkpoints/debug",
        log_dir="logs/debug",
    )

    return data_config, model_config, training_config


def get_large_model_config():
    """Get configuration for larger capacity model.

    Returns:
        Tuple of (DataConfig, ModelConfig, TrainingConfig) for large model.
    """
    data_config = DataConfig(
        batch_size=16,  # Smaller batch for larger model
    )

    model_config = ModelConfig(
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
        bidirectional=True,
        use_attention=True,
    )

    training_config = TrainingConfig(
        max_epochs=100,
        early_stopping_patience=15,
        learning_rate=5e-4,
    )

    return data_config, model_config, training_config
