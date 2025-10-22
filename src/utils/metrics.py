"""Evaluation metrics for trajectory prediction."""

import numpy as np
from typing import Optional


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis: Optional[int] = None,
) -> float:
    """Compute Root Mean Squared Error between true and predicted values.

    Args:
        y_true: Ground truth values of shape (..., 2) with x, y coordinates.
        y_pred: Predicted values with same shape as y_true.
        axis: Axis or axes along which to compute RMSE. If None, compute
            global RMSE over all elements.

    Returns:
        RMSE value (scalar if axis=None, array otherwise).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    squared_error = (y_true - y_pred) ** 2
    mean_squared_error = np.mean(squared_error, axis=axis)
    return np.sqrt(mean_squared_error)


def per_player_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute RMSE per player for detailed analysis.

    Useful for identifying which players are harder to predict or
    if certain player roles have higher prediction errors.

    Args:
        y_true: Ground truth of shape (n_players, n_frames, 2).
        y_pred: Predictions with same shape as y_true.

    Returns:
        Array of shape (n_players,) containing RMSE for each player.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    if y_true.ndim != 3:
        raise ValueError(
            f"Expected 3D input (n_players, n_frames, 2), got {y_true.ndim}D"
        )

    # Compute RMSE over frames and coordinates for each player
    squared_error = (y_true - y_pred) ** 2
    # Mean over frames and coordinates (axis 1 and 2)
    mean_squared_error = np.mean(squared_error, axis=(1, 2))
    return np.sqrt(mean_squared_error)


def per_frame_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute RMSE per frame to identify error accumulation over time.

    This metric helps identify if errors accumulate as predictions
    extend further into the future (later frames).

    Args:
        y_true: Ground truth of shape (n_samples, n_frames, 2).
        y_pred: Predictions with same shape as y_true.

    Returns:
        Array of shape (n_frames,) containing RMSE for each frame.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    if y_true.ndim != 3:
        raise ValueError(
            f"Expected 3D input (n_samples, n_frames, 2), got {y_true.ndim}D"
        )

    # Compute RMSE over samples and coordinates for each frame
    squared_error = (y_true - y_pred) ** 2
    # Mean over samples and coordinates (axis 0 and 2)
    mean_squared_error = np.mean(squared_error, axis=(0, 2))
    return np.sqrt(mean_squared_error)


def coordinate_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis: Optional[int] = None,
) -> float:
    """Compute Mean Absolute Error for coordinates.

    Alternative to RMSE that is less sensitive to outliers.

    Args:
        y_true: Ground truth values of shape (..., 2).
        y_pred: Predicted values with same shape as y_true.
        axis: Axis along which to compute MAE.

    Returns:
        MAE value.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    absolute_error = np.abs(y_true - y_pred)
    return np.mean(absolute_error, axis=axis)


def displacement_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean displacement error per prediction.

    For each (x, y) prediction, compute the Euclidean distance to the
    ground truth position.

    Args:
        y_true: Ground truth of shape (..., 2).
        y_pred: Predictions with same shape as y_true.

    Returns:
        Array of displacement errors with shape (...,).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Euclidean distance
    diff = y_true - y_pred
    return np.sqrt(np.sum(diff**2, axis=-1))


def final_displacement_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute average displacement error at the final predicted frame.

    This metric focuses on the end-point accuracy, which may be most
    important for some applications.

    Args:
        y_true: Ground truth of shape (n_samples, n_frames, 2).
        y_pred: Predictions with same shape as y_true.

    Returns:
        Average displacement error at the last frame.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Get final frame for all samples
    final_true = y_true[:, -1, :]  # (n_samples, 2)
    final_pred = y_pred[:, -1, :]  # (n_samples, 2)

    # Compute displacement
    final_displacements = displacement_error(final_true, final_pred)
    return np.mean(final_displacements)
