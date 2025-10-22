"""Dataset for NFL trajectory prediction with normalization and padding.

This module provides the PyTorch Dataset for loading parquet files,
applying coordinate normalization, and preparing sequences for seq2seq training.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from src.utils.normalization import CoordinateTransform


def normalize_features(features: np.ndarray, feature_cols: List[str]) -> np.ndarray:
    """Normalize all input features to reasonable ranges.

    Args:
        features: Array of shape (seq_len, n_features)
        feature_cols: List of feature names

    Returns:
        Normalized features with same shape
    """
    features = features.copy()

    for i, col in enumerate(feature_cols):
        if col in ["x", "y"]:
            # Already normalized by CoordinateTransform (should be ~[-0.5, 0.5])
            pass
        elif col == "s":  # Speed (0-12 yards/sec typical, clip at 15)
            features[:, i] = np.clip(features[:, i] / 15.0, 0, 1)  # Normalize to [0, 1]
        elif col == "a":  # Acceleration (-8 to 8 yards/sec^2 typical, clip at 10)
            features[:, i] = np.clip(
                features[:, i] / 10.0, -1, 1
            )  # Normalize to [-1, 1]
        elif col in ["dir", "o"]:  # Direction and orientation (0-360 degrees)
            # Already rotated by CoordinateTransform, just normalize to [-1, 1]
            features[:, i] = (
                features[:, i] % 360
            ) / 180.0 - 1.0  # Map [0, 360] to [-1, 1]

    return features


class NFLTrajectoryDataset(Dataset):
    """Dataset for NFL player trajectory prediction.

    Loads pre-processed parquet files and applies:
    - Coordinate normalization (centering, rotation, field scaling)
    - Sequence padding to fixed lengths
    - Masking for variable-length sequences

    Args:
        input_parquet: Path to input parquet file
        output_parquet: Path to output parquet file
        max_encoder_len: Fixed encoder sequence length (default: 40)
        max_decoder_len: Maximum decoder sequence length (default: 32)
        feature_cols: List of feature column names
        normalize: Whether to apply normalization
        rotation_normalize: Whether to normalize play direction
    """

    def __init__(
        self,
        input_parquet: str,
        output_parquet: str,
        max_encoder_len: int = 40,
        max_decoder_len: int = 32,
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
        rotation_normalize: bool = True,
    ):
        self.input_path = Path(input_parquet)
        self.output_path = Path(output_parquet)
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.normalize = normalize
        self.rotation_normalize = rotation_normalize

        if feature_cols is None:
            self.feature_cols = ["x", "y", "s", "a", "dir", "o"]
        else:
            self.feature_cols = feature_cols

        self.input_dim = len(self.feature_cols)
        self.output_dim = 2  # (x, y)

        # Load data
        print(f"Loading input data from {self.input_path}...")
        input_df = pd.read_parquet(self.input_path)
        print(f"Loading output data from {self.output_path}...")
        output_df = pd.read_parquet(self.output_path)

        # Pre-group data by (game_id, play_id, nfl_id) for fast access
        print("Pre-grouping data for fast access...")
        self.input_groups = {}
        self.output_groups = {}

        # Group input data
        for (game_id, play_id, nfl_id), group in input_df.groupby(
            ["game_id", "play_id", "nfl_id"]
        ):
            key = (game_id, play_id, nfl_id)
            # Sort by frame_id and extract features
            group = group.sort_values("frame_id")
            self.input_groups[key] = {
                "features": group[self.feature_cols].values.astype(np.float32),
                "play_direction": group["play_direction"].iloc[0],
            }

        # Group output data
        for (game_id, play_id, nfl_id), group in output_df.groupby(
            ["game_id", "play_id", "nfl_id"]
        ):
            key = (game_id, play_id, nfl_id)
            # Sort by frame_id and extract positions
            group = group.sort_values("frame_id")
            self.output_groups[key] = group[["x", "y"]].values.astype(np.float32)

        # Create sample index: only keys that have both input and output
        self.samples = [
            key for key in self.output_groups.keys() if key in self.input_groups
        ]

        print(f"Dataset initialized with {len(self.samples)} samples")
        print(f"  Encoder length: {self.max_encoder_len}")
        print(f"  Decoder length: {self.max_decoder_len}")
        print(f"  Features: {self.feature_cols}")
        print(f"  Normalization: {self.normalize}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dictionary containing:
                - encoder_input: (max_encoder_len, input_dim) tensor
                - decoder_input: (max_decoder_len, output_dim) tensor
                - decoder_target: (max_decoder_len, output_dim) tensor
                - encoder_mask: (max_encoder_len,) boolean tensor
                - decoder_mask: (max_decoder_len,) boolean tensor
                - metadata: dict with game_id, play_id, nfl_id
        """
        key = self.samples[idx]
        game_id, play_id, nfl_id = key

        # Get pre-grouped data (already sorted)
        input_data = self.input_groups[key]
        input_features = input_data["features"].copy()
        play_direction = input_data["play_direction"]
        output_positions = self.output_groups[key].copy()

        # Apply normalization if enabled
        if self.normalize:
            # Create transformer and fit on input
            transformer = CoordinateTransform(
                field_dims=(120.0, 53.3),
                normalize_field=self.rotation_normalize,
                normalize_rotation=self.rotation_normalize,
            )
            transformer.fit(input_features, play_direction)

            # Transform input features (handles x, y normalization and angle rotation)
            input_features = transformer.transform(input_features)

            # Normalize all features to [-1, 1] or [0, 1]
            input_features = normalize_features(input_features, self.feature_cols)

            # Transform output positions
            output_full = np.column_stack(
                [output_positions, np.zeros((len(output_positions), 4))]
            )
            output_transformed = transformer.transform(output_full)
            output_positions = output_transformed[:, :2]

        # Pad/truncate encoder sequence
        encoder_len = len(input_features)
        encoder_input = np.zeros(
            (self.max_encoder_len, self.input_dim), dtype=np.float32
        )
        encoder_mask = np.zeros(self.max_encoder_len, dtype=bool)

        if encoder_len <= self.max_encoder_len:
            # Pad on the left (keep most recent frames)
            encoder_input[-encoder_len:] = input_features
            encoder_mask[-encoder_len:] = True
        else:
            # Truncate (keep most recent frames)
            encoder_input = input_features[-self.max_encoder_len :]
            encoder_mask[:] = True

        # Pad/truncate decoder sequence
        decoder_len = len(output_positions)
        decoder_target = np.zeros(
            (self.max_decoder_len, self.output_dim), dtype=np.float32
        )
        decoder_input = np.zeros(
            (self.max_decoder_len, self.output_dim), dtype=np.float32
        )
        decoder_mask = np.zeros(self.max_decoder_len, dtype=bool)

        if decoder_len > 0:
            actual_len = min(decoder_len, self.max_decoder_len)
            decoder_target[:actual_len] = output_positions[:actual_len]
            decoder_mask[:actual_len] = True

            # Decoder input: last encoder position + shifted target (teacher forcing)
            last_encoder_pos = input_features[-1, :2]  # Last (x, y) from encoder
            decoder_input[0] = last_encoder_pos
            if actual_len > 1:
                decoder_input[1:actual_len] = output_positions[: actual_len - 1]

        # Convert to tensors
        return {
            "encoder_input": torch.from_numpy(encoder_input),
            "decoder_input": torch.from_numpy(decoder_input),
            "decoder_target": torch.from_numpy(decoder_target),
            "encoder_mask": torch.from_numpy(encoder_mask),
            "decoder_mask": torch.from_numpy(decoder_mask),
            "metadata": {
                "game_id": game_id,
                "play_id": play_id,
                "nfl_id": nfl_id,
                "encoder_len": encoder_len,
                "decoder_len": decoder_len,
            },
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched tensors
    """
    # Stack tensors
    encoder_input = torch.stack([item["encoder_input"] for item in batch])
    decoder_input = torch.stack([item["decoder_input"] for item in batch])
    decoder_target = torch.stack([item["decoder_target"] for item in batch])
    encoder_mask = torch.stack([item["encoder_mask"] for item in batch])
    decoder_mask = torch.stack([item["decoder_mask"] for item in batch])

    # Collect metadata
    metadata = [item["metadata"] for item in batch]

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "decoder_target": decoder_target,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask,
        "metadata": metadata,
    }
