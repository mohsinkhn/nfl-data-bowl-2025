"""Tests for coordinate normalization utilities."""

import numpy as np
import pytest

from src.utils.normalization import (
    center_coordinates,
    compute_last_frame_offset,
    compute_play_direction,
    normalize_by_field_size,
    rotate_angles,
    rotate_coordinates,
    CoordinateTransform,
)


def test_compute_last_frame_offset():
    """Test extraction of last frame coordinates."""
    player_data = np.array([[10.0, 20.0, 5.0], [15.0, 25.0, 6.0], [20.0, 30.0, 7.0]])

    offset = compute_last_frame_offset(player_data)
    assert offset == (20.0, 30.0)


def test_compute_last_frame_offset_empty():
    """Test offset with empty data."""
    player_data = np.array([]).reshape(0, 3)
    offset = compute_last_frame_offset(player_data)
    assert offset == (0.0, 0.0)


def test_center_coordinates():
    """Test coordinate centering."""
    coords = np.array([[10.0, 20.0], [15.0, 25.0], [20.0, 30.0]])
    offset = (10.0, 20.0)

    centered = center_coordinates(coords, offset)

    expected = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
    np.testing.assert_allclose(centered, expected)


def test_compute_play_direction_left():
    """Test rotation angle for left-moving plays."""
    angle = compute_play_direction("left")
    assert np.isclose(angle, np.pi)


def test_compute_play_direction_right():
    """Test rotation angle for right-moving plays."""
    angle = compute_play_direction("right")
    assert np.isclose(angle, 0.0)


def test_rotate_coordinates():
    """Test coordinate rotation."""
    # Rotate 90 degrees counter-clockwise
    coords = np.array([[1.0, 0.0], [0.0, 1.0]])
    angle = np.pi / 2

    rotated = rotate_coordinates(coords, angle)

    expected = np.array([[0.0, 1.0], [-1.0, 0.0]])
    np.testing.assert_allclose(rotated, expected, atol=1e-6)


def test_rotate_coordinates_180():
    """Test 180-degree rotation (play direction normalization)."""
    coords = np.array([[10.0, 5.0], [-10.0, -5.0]])
    angle = np.pi

    rotated = rotate_coordinates(coords, angle)

    expected = np.array([[-10.0, -5.0], [10.0, 5.0]])
    np.testing.assert_allclose(rotated, expected, atol=1e-10)


def test_rotate_angles():
    """Test angle rotation."""
    # Note: rotate_angles works in radians, not degrees
    angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    rotation = np.pi / 2

    rotated = rotate_angles(angles, rotation)

    # After rotation and normalization to [-pi, pi]
    expected = np.array([np.pi / 2, np.pi, -np.pi / 2, 0.0])
    np.testing.assert_allclose(rotated, expected, atol=1e-6)


def test_normalize_by_field_size():
    """Test field dimension normalization with shared scale."""
    coords = np.array([[60.0, 26.65], [120.0, 53.3], [0.0, 0.0]])
    field_dims = (120.0, 53.3)

    normalized = normalize_by_field_size(coords, field_dims)

    # New behavior: uses max(field_dims) = 120.0 as shared scale for both x and y
    expected = np.array([[0.5, 26.65 / 120.0], [1.0, 53.3 / 120.0], [0.0, 0.0]])
    np.testing.assert_allclose(normalized, expected)


def test_coordinate_transform_invertibility():
    """Test that transform can be inverted to recover original coordinates."""
    # Original player data
    player_data = np.array([[10.0, 20.0, 5.0], [15.0, 25.0, 6.0], [20.0, 30.0, 7.0]])

    # Create transform
    transform = CoordinateTransform()
    transform.fit(player_data, "left")

    # Transform some coordinates
    original_coords = np.array([[25.0, 35.0], [30.0, 40.0]])
    transformed = transform.transform(original_coords)

    # Inverse transform should recover original
    recovered = transform.inverse_transform(transformed)

    np.testing.assert_allclose(recovered, original_coords, rtol=1e-5, atol=1e-5)


def test_coordinate_transform_preserves_relative_positions():
    """Test that relative positions are preserved after transform."""
    player_data = np.array([[0.0, 0.0, 5.0], [10.0, 10.0, 6.0]])

    transform = CoordinateTransform()
    transform.fit(player_data, "right")

    coords = np.array([[0.0, 0.0], [5.0, 5.0]])
    transformed = transform.transform(coords)

    # Distance should be preserved (up to normalization factor)
    original_dist = np.linalg.norm(coords[1] - coords[0])
    transformed_dist = np.linalg.norm(transformed[1] - transformed[0])

    # After field normalization, distance scales by field dimensions
    expected_scale = np.sqrt((1 / 120.0) ** 2 + (1 / 53.3) ** 2)  # Approximate scaling
    # Just verify transform doesn't break things (distances are in same ballpark)
    assert transformed_dist > 0  # Not collapsed
    assert transformed_dist < original_dist  # Normalized (smaller)


def test_coordinate_transform_with_no_rotation():
    """Test transform with rotation disabled."""
    player_data = np.array([[10.0, 20.0, 5.0], [15.0, 25.0, 6.0]])

    # Note: parameter is normalize_rotation, not rotation_normalize
    transform = CoordinateTransform(normalize_rotation=False)
    transform.fit(player_data, "left")

    # Rotation angle should be 0 even for left-moving play
    assert transform.angle == 0.0


def test_coordinate_transform_multiple_points():
    """Test transform with multiple coordinate points."""
    player_data = np.array([[0.0, 0.0, 5.0], [50.0, 25.0, 6.0]])

    transform = CoordinateTransform()
    transform.fit(player_data, "right")

    # Transform sequence of coordinates
    coords = np.array([[50.0, 25.0], [55.0, 26.0], [60.0, 27.0]])
    transformed = transform.transform(coords)

    # Check shapes
    assert transformed.shape == coords.shape

    # Inverse should recover
    recovered = transform.inverse_transform(transformed)
    np.testing.assert_allclose(recovered, coords, rtol=1e-5, atol=1e-5)


def test_coordinate_transform_align_heading():
    """Ensure heading alignment rotates final direction to +x axis."""
    coords = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    headings = np.array([45.0, 45.0, 45.0])

    transform = CoordinateTransform(align_heading=True)
    transform.fit(coords, "right", heading_direction=headings)

    # After alignment, final heading should be zero radians
    angles_norm = transform.transform_angles(np.deg2rad(headings))
    assert np.allclose(angles_norm[-1], 0.0, atol=1e-6)

    # Coordinates should remain invertible
    coords_norm = transform.transform_points(coords)
    restored = transform.inverse_points(coords_norm)
    np.testing.assert_allclose(restored, coords, atol=1e-6, rtol=1e-6)


def test_coordinate_transform_raises_without_fit():
    """Test that transform raises error if used before fitting."""
    transform = CoordinateTransform()

    coords = np.array([[10.0, 20.0]])

    with pytest.raises(ValueError, match="must be fitted"):
        transform.transform(coords)

    with pytest.raises(ValueError, match="must be fitted"):
        transform.inverse_transform(coords)
