"""Datasets and preprocessing utilities for trajectory prediction.

This module provides a configurable data pipeline that:
1. Loads pre-processed parquet files grouped by player.
2. Applies coordinate normalization and optional augmentations.
3. Builds encoder/decoder tensors with masks.
4. Emits rich metadata for inverse transforms.

It includes two dataset classes:
    - NFLTrajectoryDataset: general-purpose loader for all players.
    - TargetReceiverTrajectoryDataset: specialized loader for targeted receivers.
Both datasets share a common `TrajectoryProcessor` to keep the transformation
logic concise and easy to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.normalization import (
    CoordinateTransform,
    rotate_coordinates,
    wrap_angle,
    normalize_weight,
    normalize_height_inches,
)
from src.utils.parsers import parse_height_to_inches

ROLE_CATEGORIES = [
    "Defensive Coverage",
    "Targeted Receiver",
    "Passer",
    "Other Route Runner",
    "Unknown",
]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RawSample:
    """Container with raw (pre-transformed) sample data."""

    key: Tuple[int, int, int]
    input_features: np.ndarray  # shape (T_enc, n_features)
    play_direction: str
    ball_land: np.ndarray  # (2,)
    output_positions: np.ndarray  # shape (T_dec, 2)
    player_role: str
    player_weight: float
    player_height_inches: float
    num_frames_output: int


@dataclass
class CanonicalSample:
    """Intermediate representation after canonical transforms."""

    features: np.ndarray  # normalized encoder features (angles in degrees)
    output_positions: np.ndarray  # canonical decoder positions
    transformer: Optional[CoordinateTransform]
    augmentation_angle: float
    last_raw_position: np.ndarray  # original coordinate space


def _load_grouped_tracking(
    input_path: Path,
    output_path: Path,
    feature_cols: List[str],
) -> Tuple[
    Dict[Tuple[int, int, int], Dict[str, Any]], Dict[Tuple[int, int, int], np.ndarray]
]:
    """Load parquet files and group them by (game_id, play_id, nfl_id)."""

    print(f"Loading input data from {input_path}...")
    input_df = pd.read_parquet(input_path)
    if "ball_angle" not in input_df.columns:
        delta_x = input_df["ball_land_x"] - input_df["x"]
        delta_y = input_df["ball_land_y"] - input_df["y"]
        input_df["ball_angle"] = np.degrees(np.arctan2(delta_y, delta_x)) % 360.0
    print(f"Loading output data from {output_path}...")
    output_df = pd.read_parquet(output_path)

    input_groups: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for key, group in input_df.groupby(["game_id", "play_id", "nfl_id"]):
        group = group.sort_values("frame_id")
        player_role_raw = group["player_role"].iloc[0]
        player_role = (
            player_role_raw if player_role_raw in ROLE_CATEGORIES else "Unknown"
        )
        input_groups[key] = {
            "features": group[feature_cols].values.astype(np.float32),
            "play_direction": group["play_direction"].iloc[0],
            "ball_land": np.array(
                [group["ball_land_x"].iloc[0], group["ball_land_y"].iloc[0]],
                dtype=np.float32,
            ),
            "num_frames_output": int(group["num_frames_output"].iloc[0]),
            "player_role": player_role,
            "player_weight": float(group["player_weight"].iloc[0]),
            "player_height_inches": parse_height_to_inches(
                group["player_height"].iloc[0]
            ),
        }

    output_groups: Dict[Tuple[int, int, int], np.ndarray] = {}
    for key, group in output_df.groupby(["game_id", "play_id", "nfl_id"]):
        output_groups[key] = group.sort_values("frame_id")[["x", "y"]].values.astype(
            np.float32
        )

    return input_groups, output_groups


# -----------------------------------------------------------------------------
# Trajectory processing core
# -----------------------------------------------------------------------------


class TrajectoryProcessor:
    """Reusable preprocessing pipeline shared by both dataset variants."""

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        *,
        max_encoder_len: int,
        max_decoder_len: int,
        normalize: bool,
        rotation_normalize: bool,
        align_heading: bool,
        rotation_augmentation_prob: float,
        rotation_augmentation_degrees: float,
        vertical_flip_prob: float,
        use_player_role: bool,
        use_player_attributes: bool,
        use_polar_targets: bool,
        use_ball_residual_targets: bool,
    ) -> None:
        self.feature_cols = feature_cols or [
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
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.normalize = normalize
        self.rotation_normalize = rotation_normalize
        self.align_heading = align_heading
        self.rotation_augmentation_prob = rotation_augmentation_prob
        self.rotation_augmentation_degrees = rotation_augmentation_degrees
        self.vertical_flip_prob = vertical_flip_prob
        self.use_player_role = use_player_role
        self.use_player_attributes = use_player_attributes
        self.use_polar_targets = use_polar_targets
        self.use_ball_residual_targets = (
            use_ball_residual_targets and not use_polar_targets
        )

        self.field_dims = (120.0, 53.3)

        # Pre-compute column indices and feature templates.
        self.dir_index = self.feature_cols.index("dir")
        self.o_index = self.feature_cols.index("o")
        self.ball_angle_index = (
            self.feature_cols.index("ball_angle")
            if "ball_angle" in self.feature_cols
            else None
        )
        self.angle_feature_names = [
            name for name in self.feature_cols if name in {"dir", "o", "ball_angle"}
        ]
        self.angle_indices = [
            self.feature_cols.index(name) for name in self.angle_feature_names
        ]

        self.base_processed_feature_cols: List[str] = []
        for col in self.feature_cols:
            if col in self.angle_feature_names:
                self.base_processed_feature_cols.extend([f"{col}_sin", f"{col}_cos"])
            else:
                self.base_processed_feature_cols.append(col)

        self.dir_sin_idx = self.base_processed_feature_cols.index("dir_sin")
        self.dir_cos_idx = self.base_processed_feature_cols.index("dir_cos")

        self.coordinate_pairs = self._compute_coordinate_pairs(self.feature_cols)
        if "ball_land_x" in self.feature_cols and "ball_land_y" in self.feature_cols:
            self.ball_land_indices = (
                self.feature_cols.index("ball_land_x"),
                self.feature_cols.index("ball_land_y"),
            )
        else:
            self.ball_land_indices = None

        self.role_to_index = {role: idx for idx, role in enumerate(ROLE_CATEGORIES)}
        self.role_vectors: Dict[str, np.ndarray] = {}
        if self.use_player_role:
            for role, idx in self.role_to_index.items():
                vec = np.zeros(len(ROLE_CATEGORIES), dtype=np.float32)
                vec[idx] = 1.0
                self.role_vectors[role] = vec
            self.role_vectors.setdefault("Unknown", self.role_vectors["Unknown"])

        self.role_feature_dim = len(ROLE_CATEGORIES) if self.use_player_role else 0
        self.attribute_feature_dim = 2 if self.use_player_attributes else 0
        self.normalized_output_dim = 1
        self.encoder_input_dim = (
            len(self.base_processed_feature_cols)
            + self.role_feature_dim
            + self.attribute_feature_dim
            + self.normalized_output_dim
        )
        self.output_dim = 2

    @staticmethod
    def _compute_coordinate_pairs(feature_cols: List[str]) -> List[Tuple[int, int]]:
        index = {name: idx for idx, name in enumerate(feature_cols)}
        pairs: List[Tuple[int, int]] = []
        if "x" in index and "y" in index:
            pairs.append((index["x"], index["y"]))
        if "ball_land_x" in index and "ball_land_y" in index:
            pairs.append((index["ball_land_x"], index["ball_land_y"]))
        return pairs

    def _rotate_coordinate_columns_inplace(
        self, features: np.ndarray, angle: float
    ) -> None:
        if angle == 0.0:
            return
        for idx_x, idx_y in self.coordinate_pairs:
            coords = rotate_coordinates(features[:, [idx_x, idx_y]], angle)
            features[:, idx_x] = coords[:, 0]
            features[:, idx_y] = coords[:, 1]

    def _vertical_flip_inplace(
        self, features: np.ndarray, output_positions: np.ndarray
    ) -> None:
        for _, idx_y in self.coordinate_pairs:
            features[:, idx_y] = -features[:, idx_y]
        output_positions[:, 1] *= -1.0
        for name in self.angle_feature_names:
            idx = self.feature_cols.index(name)
            features[:, idx] = (-features[:, idx]) % 360.0

    def _get_ball_landing(self, features: np.ndarray) -> np.ndarray:
        if self.ball_land_indices is None:
            raise ValueError("ball_land_x/ball_land_y must be present in feature columns for residual targets.")
        idx_x, idx_y = self.ball_land_indices
        return np.array([features[-1, idx_x], features[-1, idx_y]], dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, raw: RawSample) -> Dict[str, Any]:
        canonical = self._canonicalize(raw)
        encoder_input, encoder_mask, base_features = self._build_encoder_inputs(
            raw, canonical
        )
        last_position_canonical = canonical.features[-1, :2].astype(np.float32)
        last_heading = float(
            np.arctan2(
                base_features[-1, self.dir_sin_idx], base_features[-1, self.dir_cos_idx]
            )
        )

        decoder = self._build_decoder_sequences(
            canonical, last_position_canonical, last_heading
        )
        metadata = self._build_metadata(
            raw,
            canonical,
            last_position_canonical,
            last_heading,
            decoder["positions_canonical"],
        )

        return {
            "encoder_input": torch.from_numpy(encoder_input),
            "decoder_input": torch.from_numpy(decoder["decoder_input"]),
            "decoder_target": torch.from_numpy(decoder["decoder_target"]),
            "encoder_mask": torch.from_numpy(encoder_mask),
            "decoder_mask": torch.from_numpy(decoder["decoder_mask"]),
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Canonicalisation helpers
    # ------------------------------------------------------------------

    def _canonicalize(self, raw: RawSample) -> CanonicalSample:
        features = raw.input_features.astype(np.float32).copy()
        output_positions = raw.output_positions.astype(np.float32).copy()
        transformer: Optional[CoordinateTransform] = None
        augmentation_angle = 0.0
        last_raw_position = raw.input_features[-1, :2].astype(np.float32)

        if self.normalize:
            transformer = CoordinateTransform(
                field_dims=self.field_dims,
                normalize_field=True,
                normalize_rotation=self.rotation_normalize,
                align_heading=self.align_heading,
                feature_cols=self.feature_cols,
                max_speed=15.0,
                max_acceleration=10.0,
            )
            transformer.fit(
                features[:, :2],
                raw.play_direction,
                heading_direction=features[:, self.dir_index].copy(),
            )
            features = transformer.transform_feature_matrix(features, self.feature_cols)
            output_positions = transformer.transform_points(output_positions)

        if (
            self.rotation_augmentation_prob > 0.0
            and np.random.rand() < self.rotation_augmentation_prob
        ):
            max_radians = np.deg2rad(self.rotation_augmentation_degrees)
            augmentation_angle = float(np.random.uniform(-max_radians, max_radians))
            self._rotate_coordinate_columns_inplace(features, augmentation_angle)
            output_positions = rotate_coordinates(output_positions, augmentation_angle)
            features[:, self.angle_indices] = (
                features[:, self.angle_indices] + np.rad2deg(augmentation_angle)
            ) % 360.0

        if self.vertical_flip_prob > 0.0 and np.random.rand() < self.vertical_flip_prob:
            self._vertical_flip_inplace(features, output_positions)

        return CanonicalSample(
            features=features.astype(np.float32),
            output_positions=output_positions.astype(np.float32),
            transformer=transformer,
            augmentation_angle=augmentation_angle,
            last_raw_position=last_raw_position,
        )

    # ------------------------------------------------------------------
    # Encoder construction
    # ------------------------------------------------------------------

    def _build_encoder_inputs(
        self,
        raw: RawSample,
        canonical: CanonicalSample,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_features = self._convert_angle_features(canonical.features)
        blocks: List[np.ndarray] = [base_features]

        seq_len = len(base_features)

        if self.use_player_role:
            role_vec = self.role_vectors.get(
                raw.player_role, self.role_vectors.get("Unknown")
            )
            role_matrix = np.repeat(role_vec[np.newaxis, :], seq_len, axis=0)
            blocks.append(role_matrix)

        if self.use_player_attributes:
            attr_vec = np.array(
                [
                    normalize_weight(raw.player_weight),
                    normalize_height_inches(raw.player_height_inches),
                ],
                dtype=np.float32,
            )
            attr_matrix = np.repeat(attr_vec[np.newaxis, :], seq_len, axis=0)
            blocks.append(attr_matrix)

        normalized_len = min(raw.num_frames_output, self.max_decoder_len) / float(
            self.max_decoder_len
        )
        blocks.append(
            np.full((seq_len, 1), normalized_len, dtype=np.float32)
        )

        encoder_features = np.concatenate(blocks, axis=1)
        encoder_input, encoder_mask = self._pad_encoder_features(encoder_features)
        return encoder_input, encoder_mask, base_features

    def _convert_angle_features(self, features: np.ndarray) -> np.ndarray:
        columns: List[np.ndarray] = []
        for idx, name in enumerate(self.feature_cols):
            col = features[:, idx].astype(np.float32)
            if name in self.angle_feature_names:
                radians = np.deg2rad(col)
                columns.append(np.sin(radians).astype(np.float32))
                columns.append(np.cos(radians).astype(np.float32))
            else:
                columns.append(col)
        return np.stack(columns, axis=1).astype(np.float32)

    def _pad_encoder_features(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        encoder_len = len(features)
        padded = np.zeros((self.max_encoder_len, features.shape[1]), dtype=np.float32)
        mask = np.zeros(self.max_encoder_len, dtype=bool)

        if encoder_len <= self.max_encoder_len:
            padded[-encoder_len:] = features
            mask[-encoder_len:] = True
        else:
            padded = features[-self.max_encoder_len :]
            mask[:] = True

        return padded, mask

    # ------------------------------------------------------------------
    # Decoder construction
    # ------------------------------------------------------------------

    def _build_decoder_sequences(
        self,
        canonical: CanonicalSample,
        last_position_canonical: np.ndarray,
        last_heading: float,
    ) -> Dict[str, Any]:
        decoder_positions = canonical.output_positions
        decoder_len = len(decoder_positions)
        actual_len = min(decoder_len, self.max_decoder_len)

        decoder_target = np.zeros((self.max_decoder_len, 2), dtype=np.float32)
        decoder_input = np.zeros((self.max_decoder_len, 2), dtype=np.float32)
        decoder_mask = np.zeros(self.max_decoder_len, dtype=bool)
        decoder_mask[:actual_len] = True

        if actual_len == 0:
            return {
                "decoder_input": decoder_input,
                "decoder_target": decoder_target,
                "decoder_mask": decoder_mask,
                "positions_canonical": decoder_positions.astype(np.float32),
            }

        landing_canonical = None
        if self.ball_land_indices is not None:
            landing_canonical = self._get_ball_landing(canonical.features)

        if self.use_polar_targets:
            polar = self._compute_polar_deltas(
                decoder_positions, last_position_canonical, last_heading
            )
            decoder_target[:actual_len] = polar[:actual_len]
            if actual_len > 1:
                decoder_input[1:actual_len] = polar[: actual_len - 1]
        elif self.use_ball_residual_targets:
            if landing_canonical is None:
                raise ValueError("Ball landing coordinates not available for residual targets.")
            residuals = (
                landing_canonical[np.newaxis, :].astype(np.float32) - decoder_positions
            )
            decoder_target[:actual_len] = residuals[:actual_len]
            decoder_input[:actual_len] = residuals[:actual_len]
        else:
            decoder_target[:actual_len] = decoder_positions[:actual_len]
            decoder_input[0] = canonical.features[-1, :2]
            if actual_len > 1:
                decoder_input[1:actual_len] = decoder_positions[: actual_len - 1]

        return {
            "decoder_input": decoder_input,
            "decoder_target": decoder_target,
            "decoder_mask": decoder_mask,
            "positions_canonical": decoder_positions.astype(np.float32),
        }

    def _compute_polar_deltas(
        self,
        positions: np.ndarray,
        start_position: np.ndarray,
        start_heading: float,
    ) -> np.ndarray:
        deltas = np.zeros((len(positions), 2), dtype=np.float32)
        prev_pos = np.array(start_position, dtype=np.float32)
        heading = float(start_heading)
        scale = 60.0

        for idx, pos in enumerate(positions):
            delta_vec = pos - prev_pos
            step_distance = float(np.linalg.norm(delta_vec))
            delta_t = step_distance * scale
            new_heading = (
                float(np.arctan2(delta_vec[1], delta_vec[0]))
                if step_distance > 1e-8
                else heading
            )
            delta_theta = float(wrap_angle(new_heading - heading))
            heading = float(wrap_angle(new_heading))
            prev_pos = pos
            deltas[idx, 0] = delta_t
            deltas[idx, 1] = delta_theta / np.pi

        return deltas

    # ------------------------------------------------------------------
    # Metadata assembly
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        raw: RawSample,
        canonical: CanonicalSample,
        last_position_canonical: np.ndarray,
        last_heading: float,
        decoder_positions_canonical: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            "game_id": raw.key[0],
            "play_id": raw.key[1],
            "nfl_id": raw.key[2],
            "encoder_len": len(canonical.features),
            "decoder_len": len(canonical.output_positions),
            "transform": canonical.transformer,
            "ball_land_raw": raw.ball_land.astype(np.float32),
            "ball_land_canonical": (
                self._get_ball_landing(canonical.features)
                if self.ball_land_indices is not None
                else raw.ball_land.astype(np.float32)
            ),
            "augmentation_angle": float(canonical.augmentation_angle),
            "num_frames_output": int(raw.num_frames_output),
            "last_position_canonical": last_position_canonical.astype(np.float32),
            "last_heading": float(last_heading),
            "decoder_target_cartesian": decoder_positions_canonical.astype(np.float32),
            "use_polar_targets": self.use_polar_targets,
            "use_ball_residual_targets": self.use_ball_residual_targets,
            "player_role": raw.player_role,
            "player_weight": float(raw.player_weight),
            "player_height_inches": float(raw.player_height_inches),
        }


# -----------------------------------------------------------------------------
# Dataset implementations
# -----------------------------------------------------------------------------


class NFLTrajectoryDataset(Dataset):
    """General-purpose dataset covering every player."""

    def __init__(
        self,
        input_parquet: str,
        output_parquet: str,
        max_encoder_len: int = 40,
        max_decoder_len: int = 32,
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
        rotation_normalize: bool = True,
        align_heading: bool = True,
        rotation_augmentation_prob: float = 0.0,
        rotation_augmentation_degrees: float = 45.0,
        vertical_flip_prob: float = 0.0,
        use_player_role: bool = True,
        use_player_attributes: bool = True,
        use_polar_targets: bool = False,
        use_ball_residual_targets: bool = False,
    ) -> None:
        self.feature_cols = feature_cols or [
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
        self.processor = TrajectoryProcessor(
            feature_cols=self.feature_cols,
            max_encoder_len=max_encoder_len,
            max_decoder_len=max_decoder_len,
            normalize=normalize,
            rotation_normalize=rotation_normalize,
            align_heading=align_heading,
            rotation_augmentation_prob=rotation_augmentation_prob,
            rotation_augmentation_degrees=rotation_augmentation_degrees,
            vertical_flip_prob=vertical_flip_prob,
            use_player_role=use_player_role,
            use_player_attributes=use_player_attributes,
            use_polar_targets=use_polar_targets,
            use_ball_residual_targets=use_ball_residual_targets,
        )

        self.input_dim = self.processor.encoder_input_dim
        self.output_dim = self.processor.output_dim

        input_groups, output_groups = _load_grouped_tracking(
            Path(input_parquet), Path(output_parquet), self.feature_cols
        )
        common_keys = [key for key in output_groups if key in input_groups]

        self.input_groups = input_groups
        self.output_groups = output_groups
        self.samples = common_keys

        print(f"Dataset initialized with {len(self.samples)} samples")
        print(f"  Encoder length: {max_encoder_len}")
        print(f"  Decoder length: {max_decoder_len}")
        print(f"  Input dim: {self.input_dim}")
        print(f"  Output dim: {self.output_dim}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.samples[idx]
        sample = self._fetch_raw_sample(key)
        return self.processor.process(sample)

    def _fetch_raw_sample(self, key: Tuple[int, int, int]) -> RawSample:
        input_data = self.input_groups[key]
        return RawSample(
            key=key,
            input_features=input_data["features"],
            play_direction=input_data["play_direction"],
            ball_land=input_data["ball_land"],
            output_positions=self.output_groups[key],
            player_role=input_data["player_role"],
            player_weight=input_data["player_weight"],
            player_height_inches=input_data["player_height_inches"],
            num_frames_output=input_data["num_frames_output"],
        )


class TargetReceiverTrajectoryDataset(Dataset):
    """Dataset specialising in targeted receivers with optional sequence filtering."""

    def __init__(
        self,
        input_parquet: str,
        output_parquet: str,
        max_encoder_len: int = 40,
        max_decoder_len: int = 32,
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
        rotation_normalize: bool = True,
        align_heading: bool = True,
        rotation_augmentation_prob: float = 0.0,
        rotation_augmentation_degrees: float = 45.0,
        vertical_flip_prob: float = 0.0,
        use_player_attributes: bool = True,
        use_polar_targets: bool = False,
        use_ball_residual_targets: bool = False,
        min_decoder_len: int = 1,
        filter_target_only: bool = True,
    ) -> None:
        self.feature_cols = feature_cols or ["x", "y", "s", "a", "dir", "o", "ball_land_x", "ball_land_y", "ball_angle"]
        self.processor = TrajectoryProcessor(
            feature_cols=self.feature_cols,
            max_encoder_len=max_encoder_len,
            max_decoder_len=max_decoder_len,
            normalize=normalize,
            rotation_normalize=rotation_normalize,
            align_heading=align_heading,
            rotation_augmentation_prob=rotation_augmentation_prob,
            rotation_augmentation_degrees=rotation_augmentation_degrees,
            vertical_flip_prob=vertical_flip_prob,
            use_player_role=False,  # Targeted receivers do not need role encoding
            use_player_attributes=use_player_attributes,
            use_polar_targets=use_polar_targets,
            use_ball_residual_targets=use_ball_residual_targets,
        )

        self.input_dim = self.processor.encoder_input_dim
        self.output_dim = self.processor.output_dim

        input_groups, output_groups = _load_grouped_tracking(
            Path(input_parquet), Path(output_parquet), self.feature_cols
        )

        samples: List[Tuple[int, int, int]] = []
        skipped = 0
        for key, input_data in input_groups.items():
            if key not in output_groups:
                continue
            if filter_target_only and input_data["player_role"] != "Targeted Receiver":
                skipped += 1
                continue
            if len(output_groups[key]) < max(1, min_decoder_len):
                skipped += 1
                continue
            samples.append(key)

        self.input_groups = input_groups
        self.output_groups = output_groups
        self.samples = samples

        print(
            f"TargetReceiverTrajectoryDataset initialized with {len(self.samples)} samples "
            f"(filtered out {skipped})."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.samples[idx]
        sample = self._fetch_raw_sample(key)
        processed = self.processor.process(sample)
        processed["metadata"]["is_target_receiver"] = True
        return processed

    def _fetch_raw_sample(self, key: Tuple[int, int, int]) -> RawSample:
        input_data = self.input_groups[key]
        return RawSample(
            key=key,
            input_features=input_data["features"],
            play_direction=input_data["play_direction"],
            ball_land=input_data["ball_land"],
            output_positions=self.output_groups[key],
            player_role="Targeted Receiver",
            player_weight=input_data["player_weight"],
            player_height_inches=input_data["player_height_inches"],
            num_frames_output=input_data["num_frames_output"],
        )


# -----------------------------------------------------------------------------
# Collate function
# -----------------------------------------------------------------------------


def collate_fn(batch: Iterable[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Simple collate fn that stacks tensors and retains metadata."""
    batch_list = list(batch)
    encoder_input = torch.stack([item["encoder_input"] for item in batch_list])
    decoder_input = torch.stack([item["decoder_input"] for item in batch_list])
    decoder_target = torch.stack([item["decoder_target"] for item in batch_list])
    encoder_mask = torch.stack([item["encoder_mask"] for item in batch_list])
    decoder_mask = torch.stack([item["decoder_mask"] for item in batch_list])
    metadata = [item["metadata"] for item in batch_list]

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "decoder_target": decoder_target,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask,
        "metadata": metadata,
    }
