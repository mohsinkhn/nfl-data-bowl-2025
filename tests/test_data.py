import numpy as np
import pandas as pd
import torch

from src.data import NFLTrajectoryDataset, ROLE_CATEGORIES


def _write_sample_parquet(tmp_path, play_direction="left"):
    input_rows = []
    output_rows = []
    for game_id in [1]:
        for play_id in [10]:
            for nfl_id in [1001]:
                for frame_id, (x, y, s, a, direction, orient) in enumerate(
                    [
                        (30.0, 20.0, 5.0, 1.0, 45.0, 45.0),
                        (31.0, 21.0, 6.0, 1.5, 60.0, 45.0),
                        (32.0, 22.0, 7.0, 2.0, 75.0, 45.0),
                    ],
                    start=1,
                ):
                    input_rows.append(
                        {
                            "game_id": game_id,
                            "play_id": play_id,
                            "nfl_id": nfl_id,
                            "frame_id": frame_id,
                            "play_direction": play_direction,
                            "x": x,
                            "y": y,
                            "s": s,
                            "a": a,
                            "dir": direction,
                            "o": orient,
                            "ball_land_x": 40.0,
                            "ball_land_y": 30.0,
                            "num_frames_output": 3,
                            "player_role": "Targeted Receiver",
                            "player_weight": 200.0,
                            "player_height": "6-2",
                        }
                    )
                for frame_id, (x, y) in enumerate(
                    [(33.0, 23.0), (34.0, 24.0), (35.0, 25.0)],
                    start=4,
                ):
                    output_rows.append(
                        {
                            "game_id": game_id,
                            "play_id": play_id,
                            "nfl_id": nfl_id,
                            "frame_id": frame_id,
                            "x": x,
                            "y": y,
                        }
                    )

    input_df = pd.DataFrame(input_rows)
    output_df = pd.DataFrame(output_rows)
    input_path = tmp_path / "train_input.parquet"
    output_path = tmp_path / "train_output.parquet"
    input_df.to_parquet(input_path, index=False)
    output_df.to_parquet(output_path, index=False)
    return input_path, output_path


def test_dataset_pads_and_masks_sequences(tmp_path):
    input_path, output_path = _write_sample_parquet(tmp_path)
    dataset = NFLTrajectoryDataset(
        str(input_path),
        str(output_path),
        max_encoder_len=5,
        max_decoder_len=4,
    )

    sample = dataset[0]
    assert sample["encoder_input"].shape == (5, dataset.input_dim)
    assert sample["decoder_input"].shape == (4, dataset.output_dim)
    assert sample["decoder_target"].shape == (4, dataset.output_dim)

    encoder_valid = sample["encoder_mask"].sum().item()
    decoder_valid = sample["decoder_mask"].sum().item()
    assert encoder_valid == 3
    assert decoder_valid == 3

    # Decoder inputs should be shifted targets with start token
    if decoder_valid > 1:
        torch.testing.assert_close(
            sample["decoder_input"][1:decoder_valid],
            sample["decoder_target"][: decoder_valid - 1],
            atol=1e-5,
            rtol=1e-5,
        )
    torch.testing.assert_close(
        sample["decoder_input"][0],
        sample["encoder_input"][-1, :2],
        atol=1e-5,
        rtol=1e-5,
    )

    # Get mask indices first before using them
    encoder_input = sample["encoder_input"].numpy()
    mask_indices = np.nonzero(sample["encoder_mask"].numpy())[0]

    dir_sin_idx = dataset.final_feature_cols.index("dir_sin")
    dir_cos_idx = dataset.final_feature_cols.index("dir_cos")
    last_dir_sin = sample["encoder_input"][mask_indices[-1], dir_sin_idx].item()
    last_dir_cos = sample["encoder_input"][mask_indices[-1], dir_cos_idx].item()
    dir_norm = np.sqrt(last_dir_sin**2 + last_dir_cos**2)
    assert np.isclose(dir_norm, 1.0, atol=1e-5)

    role_start_idx = dataset.final_feature_cols.index("role_defensive_coverage")
    role_end_idx = role_start_idx + len(ROLE_CATEGORIES)
    role_vector = sample["encoder_input"][mask_indices[-1], role_start_idx:role_end_idx].numpy()
    expected_role = np.zeros(len(ROLE_CATEGORIES), dtype=np.float32)
    expected_role[ROLE_CATEGORIES.index("Targeted Receiver")] = 1.0
    np.testing.assert_allclose(role_vector, expected_role, atol=1e-5, rtol=1e-5)

    weight_idx = dataset.final_feature_cols.index("player_weight_norm")
    height_idx = dataset.final_feature_cols.index("player_height_norm")
    weight_norm = sample["encoder_input"][mask_indices[-1], weight_idx].item()
    height_norm = sample["encoder_input"][mask_indices[-1], height_idx].item()
    assert np.isclose(weight_norm, 200.0 / 400.0, atol=1e-5)
    assert np.isclose(height_norm, (6 * 12 + 2) / 80.0, atol=1e-5)

    # Derived landing vector features should align with canonical landing vector
    derived_count = len(dataset.derived_feature_names)
    landing_dx_idx = dataset.input_dim - derived_count
    landing_dy_idx = landing_dx_idx + 1
    landing_angle_idx = landing_dy_idx + 1
    normalized_output_idx = landing_angle_idx + 1
    dir_sin_idx = dataset.final_feature_cols.index("dir_sin")
    dir_cos_idx = dataset.final_feature_cols.index("dir_cos")

    metadata = sample["metadata"]
    key = (metadata["game_id"], metadata["play_id"], metadata["nfl_id"])
    transform = metadata["transform"]
    raw_features = dataset.input_groups[key]["features"]
    last_raw = raw_features[-1, :2]
    ball_raw = metadata["ball_land_raw"]
    transformed_points = transform.transform_points(
        np.stack([last_raw, ball_raw], axis=0)
    )
    expected_vec = transformed_points[1]

    # Only check the last valid frame where the landing vector assertion makes sense
    idx = mask_indices[-1]
    dx = encoder_input[idx, landing_dx_idx]
    dy = encoder_input[idx, landing_dy_idx]
    np.testing.assert_allclose([dx, dy], expected_vec, atol=1e-5, rtol=1e-5)

    dir_sin = encoder_input[idx, dir_sin_idx]
    dir_cos = encoder_input[idx, dir_cos_idx]
    dir_vec = np.array([dir_cos, dir_sin])
    dir_unit = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
    vec_unit = expected_vec / (np.linalg.norm(expected_vec) + 1e-8)
    cross_val = dir_unit[0] * vec_unit[1] - dir_unit[1] * vec_unit[0]
    dot_val = np.clip(dir_unit[0] * vec_unit[0] + dir_unit[1] * vec_unit[1], -1.0, 1.0)
    expected_angle = np.arctan2(cross_val, dot_val) / np.pi
    np.testing.assert_allclose(
        expected_angle,
        encoder_input[idx, landing_angle_idx],
        atol=1e-5,
        rtol=1e-5,
    )

    normalized_len = encoder_input[idx, normalized_output_idx]
    assert np.isclose(normalized_len, 3 / dataset.max_decoder_len, atol=1e-5)

    assert "last_heading" in metadata
    assert isinstance(metadata["last_heading"], float)
    assert "decoder_target_cartesian" in metadata
    assert metadata["decoder_target_cartesian"].shape[1] == 2


def test_rotation_normalizes_orientation(tmp_path):
    input_path, output_path = _write_sample_parquet(tmp_path, play_direction="left")
    dataset_rot = NFLTrajectoryDataset(
        str(input_path),
        str(output_path),
        max_encoder_len=4,
        max_decoder_len=3,
        align_heading=False,
    )
    dataset_no_rot = NFLTrajectoryDataset(
        str(input_path),
        str(output_path),
        max_encoder_len=4,
        max_decoder_len=3,
        rotation_normalize=False,
        align_heading=False,
    )

    sample_rot = dataset_rot[0]
    sample_no_rot = dataset_no_rot[0]

    orient_sin_idx = dataset_rot.final_feature_cols.index("o_sin")
    orient_cos_idx = dataset_rot.final_feature_cols.index("o_cos")

    mask_rot = np.nonzero(sample_rot["encoder_mask"].numpy())[0]
    mask_no_rot = np.nonzero(sample_no_rot["encoder_mask"].numpy())[0]
    last_idx_rot = mask_rot[-1]
    last_idx_no_rot = mask_no_rot[-1]

    orient_rot = sample_rot["encoder_input"][
        last_idx_rot, [orient_cos_idx, orient_sin_idx]
    ].numpy()
    orient_no_rot = sample_no_rot["encoder_input"][
        last_idx_no_rot, [orient_cos_idx, orient_sin_idx]
    ].numpy()

    assert not np.allclose(orient_rot, orient_no_rot, atol=1e-5)
    np.testing.assert_allclose(
        orient_rot,
        -orient_no_rot,
        atol=1e-5,
        rtol=1e-5,
    )
