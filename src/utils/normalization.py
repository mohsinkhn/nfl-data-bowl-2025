"""Coordinate normalization and transformation utilities for NFL trajectory data.

This module provides functions to normalize player coordinates and orientations
to a canonical reference frame. The main transformations are:
1. Centering: Last pre-throw frame at origin (0, 0)
2. Rotation: Play direction normalized to left-to-right
3. Field normalization: Scale by field dimensions (120 x 53.3 yards)

All transformations are invertible to convert predictions back to original coordinates.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


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
        Rotated angles, normalized to [-pi, pi].
    """
    angles = angles.copy()
    angles = angles + rotation

    # Normalize to [-pi, pi]
    angles = np.arctan2(np.sin(angles), np.cos(angles))

    return angles


def normalize_by_field_size(
    coords: np.ndarray,
    field_dims: Tuple[float, float] = (120.0, 53.3),
) -> np.ndarray:
    """Normalize coordinates by field dimensions.

    Args:
        coords: Array of shape (..., 2) with x, y coordinates in last dimension.
        field_dims: Tuple of (field_length, field_width) in yards.
            Default is (120, 53.3) for NFL field.

    Returns:
        Normalized coordinates in range approximately [-1, 1].
    """
    coords = coords.copy()
    coords[..., 0] /= field_dims[0]
    coords[..., 1] /= field_dims[1]
    return coords


def denormalize_predictions(
    preds: np.ndarray,
    offset: Tuple[float, float],
    angle: float,
    field_dims: Tuple[float, float] = (120.0, 53.3),
) -> np.ndarray:
    """Reverse all normalization transforms for final predictions.

    Applies inverse transformations in reverse order:
    1. Denormalize by field size
    2. Rotate back to original play direction
    3. Add offset to restore original position

    Args:
        preds: Predictions of shape (..., 2) with normalized x, y coordinates.
        offset: Original (x_offset, y_offset) used for centering.
        angle: Original rotation angle in radians.
        field_dims: Field dimensions (length, width) in yards.

    Returns:
        Denormalized predictions in original coordinate system.
    """
    preds = preds.copy()

    # Step 1: Denormalize by field size
    preds[..., 0] *= field_dims[0]
    preds[..., 1] *= field_dims[1]

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
    3. Normalize by field size

    The transform parameters are stored to enable inverse transformation
    of predictions back to original coordinates.

    Example:
        >>> transform = CoordinateTransform()
        >>> transform.fit(player_data, play_direction='left')
        >>> normalized = transform.transform(player_data)
        >>> # ... model prediction ...
        >>> original = transform.inverse_transform(predictions)
    """

    def __init__(
        self,
        field_dims: Tuple[float, float] = (120.0, 53.3),
        normalize_field: bool = True,
        normalize_rotation: bool = True,
    ):
        """Initialize coordinate transform.

        Args:
            field_dims: Field dimensions (length, width) in yards.
            normalize_field: Whether to normalize by field dimensions.
            normalize_rotation: Whether to rotate to canonical direction.
        """
        self.field_dims = field_dims
        self.normalize_field = normalize_field
        self.normalize_rotation = normalize_rotation

        self.offset: Optional[Tuple[float, float]] = None
        self.angle: Optional[float] = None
        self.fitted: bool = False

    def fit(
        self,
        player_data: np.ndarray,
        play_direction: str,
    ) -> "CoordinateTransform":
        """Compute transformation parameters from data.

        Args:
            player_data: Array of shape (n_frames, n_features) where first two
                columns are x, y coordinates.
            play_direction: 'left' or 'right' indicating play direction.

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

        self.fitted = True
        return self

    def transform(
        self,
        data: np.ndarray,
        angle_cols: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply normalization transform to data.

        Args:
            data: Array of shape (..., n_features) where first two columns
                are x, y coordinates.
            angle_cols: Optional array of shape (..., n_angles) containing
                orientation/direction angles to rotate.

        Returns:
            Transformed data with same shape as input.
        """
        if not self.fitted:
            raise ValueError("Transform must be fitted before use. Call fit() first.")

        data = data.copy()

        # Extract coordinates (first 2 columns)
        coords = data[..., :2]

        # Apply centering
        coords = center_coordinates(coords, self.offset)

        # Apply rotation
        if self.normalize_rotation:
            coords = rotate_coordinates(coords, self.angle)

        # Apply field normalization
        if self.normalize_field:
            coords = normalize_by_field_size(coords, self.field_dims)

        # Update coordinates in data
        data[..., :2] = coords

        # Rotate angle columns if provided
        if angle_cols is not None and self.normalize_rotation:
            angle_cols = rotate_angles(angle_cols, self.angle)

        return data

    def inverse_transform(
        self,
        preds: np.ndarray,
        angle_cols: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply inverse transform to predictions.

        Args:
            preds: Predictions of shape (..., 2) with normalized x, y coordinates.
            angle_cols: Optional array of shape (..., n_angles) containing
                predicted angles to rotate back.

        Returns:
            Denormalized predictions in original coordinate system.
        """
        if not self.fitted:
            raise ValueError("Transform must be fitted before use. Call fit() first.")

        preds = denormalize_predictions(
            preds,
            self.offset,
            self.angle if self.normalize_rotation else 0.0,
            self.field_dims if self.normalize_field else (1.0, 1.0),
        )

        # Rotate angle columns back if provided
        if angle_cols is not None and self.normalize_rotation:
            angle_cols = rotate_angles(angle_cols, -self.angle)

        return preds

    def get_params(self) -> Dict[str, Any]:
        """Get transformation parameters for serialization.

        Returns:
            Dictionary containing offset, angle, field_dims, and flags.
        """
        return {
            "offset": self.offset,
            "angle": self.angle,
            "field_dims": self.field_dims,
            "normalize_field": self.normalize_field,
            "normalize_rotation": self.normalize_rotation,
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
        self.field_dims = params.get("field_dims", (120.0, 53.3))
        self.normalize_field = params.get("normalize_field", True)
        self.normalize_rotation = params.get("normalize_rotation", True)
        self.fitted = params.get("fitted", False)
        return self
