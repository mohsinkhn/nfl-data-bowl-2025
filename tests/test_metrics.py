import numpy as np

from src.utils import metrics


def test_rmse_matches_expected():
    y_true = np.array([[0.0, 0.0], [3.0, 4.0]])
    y_pred = np.array([[0.0, 0.0], [0.0, 0.0]])
    result = metrics.rmse(y_true, y_pred)
    # First point: error = 0, Second point: error_x=3, error_y=4
    # Total squared error = 0 + 9 + 16 = 25
    # Mean over 4 elements (2 points * 2 coords) = 25/4 = 6.25
    # RMSE = sqrt(6.25) = 2.5
    assert np.isclose(result, 2.5)


def test_per_player_rmse_shape():
    y_true = np.zeros((3, 2, 2))
    y_pred = np.ones((3, 2, 2))
    result = metrics.per_player_rmse(y_true, y_pred)
    assert result.shape == (3,)
    # Each player has 2 frames * 2 coords = 4 elements
    # Each element has squared error of 1
    # Mean squared error = 1, RMSE = 1.0
    assert np.allclose(result, 1.0)


def test_final_displacement_error():
    y_true = np.array(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [2.0, 0.0]],
        ]
    )
    y_pred = np.array(
        [
            [[0.0, 0.0], [1.0, 2.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ]
    )
    # Final frame errors: sqrt((0)^2 + (1)^2) = 1, sqrt((1)^2 + 0) = 1 -> mean = 1
    assert np.isclose(metrics.final_displacement_error(y_true, y_pred), 1.0)
