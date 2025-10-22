"""Configuration presets for Approach 1."""

from .config_approach1_v0 import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    get_default_config,
    get_fast_debug_config,
    get_large_model_config,
)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "get_default_config",
    "get_fast_debug_config",
    "get_large_model_config",
]
