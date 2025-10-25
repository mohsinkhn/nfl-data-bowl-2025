"""Coordinate normalization and transformation utilities for NFL trajectory data.

This module provides functions to normalize player coordinates and orientations
to a canonical reference frame. The main transformations are:
1. Centering: Last pre-throw frame at origin (0, 0)
2. Rotation: Play direction normalized to left-to-right
3. Field normalization: Scale by field dimensions (120 x 53.3 yards)

All transformations are invertible to convert predictions back to original coordinates.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Union, List


def normalize_speed(
    speed: Union[float, np.ndarray], max_speed: float = 15.0
) -> Union[float, np.ndarray]:
    """Normalize speed (yards/sec) to [0, 1] with clipping."""
    return np.clip(np.asarray(speed, dtype=np.float32) / max_speed, 0.0, 1.0)


def normalize_acceleration(
    acceleration: Union[float, np.ndarray], max_acc: float = 10.0
) -> Union[float, np.ndarray]:
    """Normalize acceleration (yards/sec^2) to [-1, 1] with clipping."""
    return np.clip(np.asarray(acceleration, dtype=np.float32) / max_acc, -1.0, 1.0)


def normalize_weight(
    weight: Union[float, np.ndarray], max_weight: float = 400.0
) -> Union[float, np.ndarray]:
    """Normalize player weight to [0, 1] (capped at ``max_weight``)."""
    weight_arr = np.asarray(weight, dtype=np.float32)
    weight_arr = np.clip(weight_arr, 0.0, max_weight)
    return weight_arr / max_weight


def normalize_height_inches(
    height_inches: Union[float, np.ndarray], max_height: float = 80.0
) -> Union[float, np.ndarray]:
    """Normalize player height in inches to [0, 1] (capped at ``max_height``)."""
    height_arr = np.asarray(height_inches, dtype=np.float32)
    height_arr = np.clip(height_arr, 0.0, max_height)
    return height_arr / max_height


def wrap_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle(s) to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def reconstruct_trajectory_from_polar(
    deltas: np.ndarray,
    start_position: np.ndarray,
    start_heading: float,
) -> np.ndarray:
    """Integrate polar deltas (Δt, Δθ/π) into canonical coordinates."""
    if deltas.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    positions = []
    pos = np.array(start_position, dtype=np.float32)
    heading = float(start_heading)

    scale = 60.0
    for delta_t, delta_theta_norm in deltas:
        delta_angle = float(delta_theta_norm) * np.pi
        heading = float(wrap_angle(heading + delta_angle))
        step = np.array(
            [
                float(delta_t) / scale * np.cos(heading),
                float(delta_t) / scale * np.sin(heading),
            ],
            dtype=np.float32,
        )
        pos = pos + step
        positions.append(pos.copy())

    return np.stack(positions, axis=0)


def compute_last_frame_offset(
    player_data: np.ndarray,
) -> Tuple[float, float]:
    """Extract (x, y) coordinates of the last frame for centering.

    Args:
        player_data: Array of shape (n_frames, n_features) where first two
            columns are x and y coordinates.

    Returns:
        Tuple of (x_offset, y_offset) from the last frame.
    """
    if len(player_data) == 0:
        return (0.0, 0.0)

    last_frame = player_data[-1]
    x_offset = float(last_frame[0])
    y_offset = float(last_frame[1])

    return (x_offset, y_offset)


def center_coordinates(
    coords: np.ndarray,
    offset: Tuple[float, float],
) -> np.ndarray:
    """Center coordinates by subtracting offset.

    Args:
        coords: Array of shape (..., 2) with x, y coordinates in last dimension.
        offset: Tuple of (x_offset, y_offset) to subtract.

    Returns:
        Centered coordinates with same shape as input.
    """
    coords = coords.copy()
    coords[..., 0] -= offset[0]
    coords[..., 1] -= offset[1]
    return coords


def compute_play_direction(
    play_direction: str,
) -> float:
    """Compute rotation angle to normalize play direction to left-to-right.

    The field coordinate system has plays moving either 'left' or 'right'.
    We normalize all plays to move from left to right (positive x direction).

    Args:
        play_direction: Either 'left' or 'right' indicating play direction.

    Returns:
        Rotation angle in radians. 0 if already going right, pi if going left.
    """
    if play_direction.lower() == "left":
        # Rotate 180 degrees to flip left-to-right
        return np.pi
    else:
        # Already going right, no rotation needed
        return 0.0


def rotate_coordinates(
    coords: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate (x, y) coordinates by angle around origin.

    Uses standard 2D rotation matrix:
    [x']   [cos(θ)  -sin(θ)] [x]
    [y'] = [sin(θ)   cos(θ)] [y]

    Args:
        coords: Array of shape (..., 2) with x, y coordinates in last dimension.
        angle: Rotation angle in radians (counter-clockwise).

    Returns:
        Rotated coordinates with same shape as input.
    """
    coords = coords.copy()
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Store original values to avoid in-place modification issues
    x = coords[..., 0].copy()
    y = coords[..., 1].copy()

    coords[..., 0] = cos_angle * x - sin_angle * y
    coords[..., 1] = sin_angle * x + cos_angle * y

    return coords


def rotate_angles(
    angles: np.ndarray,
    rotation: float,
) -> np.ndarray:
    """Rotate orientation/direction angles by adding rotation offset.

    Args:
        angles: Array of angles in radians.
        rotation: Rotation angle in radians to add.

    Returns:
        Rotated angles, normalized to canonical range [-pi, pi].
    """
    angles = angles.copy()
    angles = angles + rotation

    # Normalize to canonical range [-pi, pi]
    angles = np.arctan2(np.sin(angles), np.cos(angles))

    return angles


def normalize_by_field_size(
    coords: np.ndarray,
    field_dims: Tuple[float, float] = (120.0, 53.3),
) -> np.ndarray:
    """Normalize coordinates by a single shared scale derived from field size.

    Args:
        coords: Array of shape (..., 2) with x, y coordinates in last dimension.
        field_dims: Tuple of (field_length, field_width) in yards.

    Returns:
        Coordinates scaled by the maximum field dimension so x and y share the same scale.
    """
    coords = coords.copy()
    scale = max(field_dims)
    coords /= scale
    return coords


def denormalize_predictions(
    preds: np.ndarray,
    offset: Tuple[float, float],
    angle: float,
    field_dims: Tuple[float, float] = (120.0, 53.3),
    heading_angle: float = 0.0,
    align_heading: bool = False,
) -> np.ndarray:
    """Reverse all normalization transforms for final predictions.

    Applies inverse transformations in reverse order:
    1. Denormalize by field size
    2. Rotate back to original play direction
    3. Add offset to restore original position
    4. Optionally undo heading alignment

    Args:
        preds: Predictions of shape (..., 2) with normalized x, y coordinates.
        offset: Original (x_offset, y_offset) used for centering.
        angle: Original rotation angle in radians.
        field_dims: Field dimensions (length, width) in yards.
        heading_angle: Heading rotation (radians) applied during normalization.
        align_heading: Whether heading alignment was active.

    Returns:
        Denormalized predictions in original coordinate system.
    """
    preds = preds.copy()

    # Step 1: Denormalize by shared field size scale
    scale = max(field_dims)
    preds *= scale

    if align_heading and heading_angle:
        preds = rotate_coordinates(preds, heading_angle)

    # Step 2: Rotate back (inverse rotation = negative angle)
    preds = rotate_coordinates(preds, -angle)

    # Step 3: Add offset back
    preds[..., 0] += offset[0]
    preds[..., 1] += offset[1]

    return preds


class CoordinateTransform:
    """Stateful coordinate transform that stores parameters for reversibility.

    This class encapsulates the full normalization pipeline:
    1. Center on last frame
    2. Rotate to canonical direction
    3. Optionally align heading so final direction points along +x
    4. Normalize by field size

    The transform parameters are stored to enable inverse transformation
    of predictions back to original coordinates.

    Example:
        >>> transform = CoordinateTransform()
        >>> transform.fit(player_data, play_direction='left')
        >>> normalized = transform.transform_points(player_data[:, :2])
        >>> # ... model prediction ...
        >>> original = transform.inverse_points(normalized)
    """

    def __init__(
        self,
        field_dims: Tuple[float, float] = (120.0, 53.3),
        normalize_field: bool = True,
        normalize_rotation: bool = True,
        align_heading: bool = False,
        feature_cols: Optional[List[str]] = None,
        max_speed: float = 15.0,
        max_acceleration: float = 10.0,
    ):
        """Initialize coordinate transform.

        Args:
            field_dims: Field dimensions (length, width) in yards.
            normalize_field: Whether to normalize by field dimensions.
            normalize_rotation: Whether to rotate to canonical direction.
            align_heading: Whether to align heading so final direction is +x.
        """
        self.field_dims = field_dims
        self.normalize_field = normalize_field
        self.normalize_rotation = normalize_rotation
        self.align_heading = align_heading
        self.feature_cols = feature_cols
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration

        self.offset: Optional[Tuple[float, float]] = None
        self.angle: Optional[float] = None
        self.heading_angle: Optional[float] = None
        self.fitted: bool = False

        self._speed_index: Optional[int] = None
        self._acc_index: Optional[int] = None
        self._angle_indices: List[int] = []
        if feature_cols is not None:
            if "s" in feature_cols:
                self._speed_index = feature_cols.index("s")
            if "a" in feature_cols:
                self._acc_index = feature_cols.index("a")
            self._angle_indices = [
                idx for idx, name in enumerate(feature_cols) if name in {"dir", "o", "ball_angle"}
            ]

    def fit(
        self,
        player_data: np.ndarray,
        play_direction: str,
        heading_direction: Optional[np.ndarray] = None,
    ) -> "CoordinateTransform":
        """Compute transformation parameters from data.

        Args:
            player_data: Array of shape (n_frames, n_features) where first two
                columns are x, y coordinates.
            play_direction: 'left' or 'right' indicating play direction.
            heading_direction: Optional array (n_frames,) of heading angles in degrees.

        Returns:
            Self for chaining.
        """
        # Compute centering offset from last frame
        self.offset = compute_last_frame_offset(player_data)

        # Compute rotation angle
        if self.normalize_rotation:
            self.angle = compute_play_direction(play_direction)
        else:
            self.angle = 0.0

        # Optional heading alignment (after play direction normalization)
        self.heading_angle = 0.0
        if (
            self.align_heading
            and heading_direction is not None
            and len(heading_direction) > 0
        ):
            # Convert final heading to radians and apply play-direction rotation
            final_heading = float(heading_direction[-1])
            heading_rad = np.deg2rad(final_heading)
            if self.normalize_rotation and self.angle:
                heading_rad = rotate_angles(np.array([heading_rad]), self.angle)[0]
            self.heading_angle = heading_rad

        self.fitted = True
        return self

    def transform_points(self, coords: np.ndarray) -> np.ndarray:
        """Apply normalization pipeline to raw coordinates."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use. Call fit() first.")

        coords = coords.copy()
        coords = center_coordinates(coords, self.offset)

        if self.normalize_rotation and self.angle:
            coords = rotate_coordinates(coords, self.angle)

        if self.align_heading and self.heading_angle:
            coords = rotate_coordinates(coords, -self.heading_angle)

        if self.normalize_field:
            coords = normalize_by_field_size(coords, self.field_dims)

        return coords

    def transform_feature_matrix(
        self,
        features: np.ndarray,
        feature_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Apply coordinate/direction/dynamics normalization to full feature matrix."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use. Call fit() first.")

        feature_cols = feature_cols or self.feature_cols
        if feature_cols is None:
            raise ValueError("feature_cols must be provided when not set on transform.")

        transformed = features.copy()
        index = {name: idx for idx, name in enumerate(feature_cols)}

        input_coords = transformed[:, :2]
        transformed[:, :2] = self.transform_points(input_coords)

        if "ball_land_x" in index and "ball_land_y" in index:
            ball_coords = np.stack(
                [
                    transformed[:, index["ball_land_x"]],
                    transformed[:, index["ball_land_y"]],
                ],
                axis=1,
            )
            ball_coords = self.transform_points(ball_coords)
            transformed[:, index["ball_land_x"]] = ball_coords[:, 0]
            transformed[:, index["ball_land_y"]] = ball_coords[:, 1]

        for idx in self._angle_indices:
            radians = np.deg2rad(transformed[:, idx])
            rotated = self.transform_angles(radians)
            transformed[:, idx] = np.rad2deg(rotated) % 360.0

        if self._speed_index is not None:
            idx = self._speed_index
            transformed[:, idx] = normalize_speed(transformed[:, idx], self.max_speed)

        if self._acc_index is not None:
            idx = self._acc_index
            transformed[:, idx] = normalize_acceleration(
                transformed[:, idx], self.max_acceleration
            )

        return transformed

    def inverse_points(self, coords: np.ndarray) -> np.ndarray:
        """Invert normalization pipeline for coordinates."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use. Call fit() first.")

        coords = coords.copy()

        if self.normalize_field:
            # Use shared scale (max of field dims) to match transform_points
            scale = max(self.field_dims)
            coords *= scale

        if self.align_heading and self.heading_angle:
            coords = rotate_coordinates(coords, self.heading_angle)

        if self.normalize_rotation and self.angle:
            coords = rotate_coordinates(coords, -self.angle)

        coords[..., 0] += self.offset[0]
        coords[..., 1] += self.offset[1]

        return coords

    def transform_angles(self, angles: np.ndarray) -> np.ndarray:
        """Rotate angles (in radians) with the same pipeline as coordinates."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use. Call fit() first.")

        rotated = angles.copy()
        if self.normalize_rotation and self.angle:
            rotated = rotate_angles(rotated, self.angle)
        if self.align_heading and self.heading_angle:
            rotated = rotate_angles(rotated, -self.heading_angle)
        return rotated

    def inverse_angles(self, angles: np.ndarray) -> np.ndarray:
        """Undo rotation pipeline for angles (in radians)."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use. Call fit() first.")

        restored = angles.copy()
        if self.align_heading and self.heading_angle:
            restored = rotate_angles(restored, self.heading_angle)
        if self.normalize_rotation and self.angle:
            restored = rotate_angles(restored, -self.angle)
        return restored

    def get_params(self) -> Dict[str, Any]:
        """Get transformation parameters for serialization.

        Returns:
            Dictionary containing offset, angle, field_dims, and flags.
        """
        return {
            "offset": self.offset,
            "angle": self.angle,
            "heading_angle": self.heading_angle,
            "field_dims": self.field_dims,
            "normalize_field": self.normalize_field,
            "normalize_rotation": self.normalize_rotation,
            "align_heading": self.align_heading,
            "fitted": self.fitted,
        }

    def set_params(self, params: Dict[str, Any]) -> "CoordinateTransform":
        """Set transformation parameters from dictionary.

        Args:
            params: Dictionary containing transformation parameters.

        Returns:
            Self for chaining.
        """
        self.offset = params.get("offset")
        self.angle = params.get("angle")
        self.heading_angle = params.get("heading_angle")
        self.field_dims = params.get("field_dims", (120.0, 53.3))
        self.normalize_field = params.get("normalize_field", True)
        self.normalize_rotation = params.get("normalize_rotation", True)
        self.align_heading = params.get("align_heading", False)
        self.fitted = params.get("fitted", False)
        return self
