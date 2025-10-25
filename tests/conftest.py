"""Shared pytest fixtures and configuration."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_input_data():
    """Create sample input data for testing."""
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2, 2],
            "play_id": [10, 10, 10, 20, 20, 20],
            "nfl_id": [1001, 1001, 1001, 1002, 1002, 1002],
            "frame_id": [1, 2, 3, 1, 2, 3],
            "play_direction": ["right", "right", "right", "left", "left", "left"],
            "x": [10.0, 15.0, 20.0, 100.0, 95.0, 90.0],
            "y": [20.0, 25.0, 30.0, 30.0, 25.0, 20.0],
            "s": [5.0, 6.0, 7.0, 5.5, 6.5, 7.5],
            "a": [1.0, 1.5, 2.0, 1.2, 1.7, 2.2],
            "dir": [45.0, 50.0, 55.0, 225.0, 230.0, 235.0],
            "o": [90.0, 95.0, 100.0, 270.0, 275.0, 280.0],
        }
    )


@pytest.fixture
def sample_output_data():
    """Create sample output data for testing."""
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2, 2],
            "play_id": [10, 10, 10, 20, 20, 20],
            "nfl_id": [1001, 1001, 1001, 1002, 1002, 1002],
            "frame_id": [4, 5, 6, 4, 5, 6],
            "x": [25.0, 30.0, 35.0, 85.0, 80.0, 75.0],
            "y": [35.0, 40.0, 45.0, 15.0, 10.0, 5.0],
        }
    )


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    import torch

    torch.manual_seed(42)
    return 42
