"""Evaluate a trained model checkpoint on validation data."""

import argparse
import torch
import numpy as np
from pathlib import Path

from src.configs.config_approach1_v0 import DataConfig, ModelConfig, TrainingConfig
from src.utils.normalization import reconstruct_trajectory_from_polar
from src.data import NFLTrajectoryDataset, collate_fn
from src.trainer import TrajectoryPredictionModule
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_model(checkpoint_path: str, fold: str = "fold1", batch_size: int = 32):
    """Evaluate model on validation set.

    Args:
        checkpoint_path: Path to model checkpoint
        fold: Which fold to evaluate
        batch_size: Batch size for evaluation
    """
    print(f"\n{'='*60}")
    print("Model Evaluation on Validation Set")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Fold: {fold}")

    # Load checkpoint to get configs
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    data_config = DataConfig(**checkpoint["hyper_parameters"]["data_config"])
    model_config = ModelConfig(**checkpoint["hyper_parameters"]["model_config"])
    training_config = TrainingConfig(**checkpoint["hyper_parameters"]["training_config"])

    print(f"✓ Configs loaded")
    print(f"  Encoder length: {data_config.max_encoder_len}")
    print(f"  Decoder length: {data_config.max_decoder_len}")

    # Load validation dataset
    print(f"\nLoading validation dataset...")
    val_dataset = NFLTrajectoryDataset(
        input_parquet=f"data/processed/{fold}/val_input.parquet",
        output_parquet=f"data/processed/{fold}/val_output.parquet",
        max_encoder_len=data_config.max_encoder_len,
        max_decoder_len=data_config.max_decoder_len,
        feature_cols=data_config.feature_cols,
        normalize=data_config.normalize,
        rotation_normalize=data_config.rotation_normalize,
        align_heading=getattr(data_config, "align_heading", True),
        use_player_role=getattr(data_config, "use_player_role", True),
        use_player_attributes=getattr(data_config, "use_player_attributes", True),
        use_polar_targets=getattr(data_config, "use_polar_targets", False),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Load model
    print(f"\nLoading model...")
    model_config.input_dim = val_dataset.input_dim
    model_config.output_dim = val_dataset.output_dim

    model = TrajectoryPredictionModule.load_from_checkpoint(
        checkpoint_path,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
    )
    model.eval()
    model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"✓ Model loaded on {device}")

    # Evaluate
    print(f"\nRunning evaluation on {len(val_dataset):,} samples...")

    all_predictions = []
    all_targets = []
    all_masks = []
    all_transforms = []
    use_polar = getattr(data_config, "use_polar_targets", False)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_target = batch["decoder_target"].to(device)
            decoder_mask = batch["decoder_mask"]
            metadata = batch["metadata"]

            prediction_mode = getattr(model, "prediction_mode", "autoregressive")
            max_len = int(decoder_mask.sum(dim=1).max().item())
            if prediction_mode == "autoregressive":
                decoder_start = decoder_input[:, 0, :]  # (batch, 2)
                predictions = model.model.predict(
                    encoder_inputs=encoder_input,
                    decoder_start=decoder_start,
                    prediction_steps=int(max_len),
                )
            else:
                predictions = model.model.predict(
                    encoder_inputs=encoder_input,
                    encoder_mask=batch.get("encoder_mask"),
                )
                if predictions.size(1) > max_len:
                    predictions = predictions[:, :max_len]

            # Store for metrics (already at max_decoder_len)
            all_predictions.append(predictions.cpu())
            all_targets.append(decoder_target.cpu())
            all_masks.append(decoder_mask)
            all_transforms.extend(metadata)

    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0).numpy()  # (N, T, 2)
    targets = torch.cat(all_targets, dim=0).numpy()  # (N, T, 2)
    masks = torch.cat(all_masks, dim=0).numpy()  # (N, T)

    predictions_canonical = np.zeros_like(predictions)
    targets_canonical = np.zeros_like(targets)
    for i, meta in enumerate(all_transforms):
        mask_i = masks[i].astype(bool)
        if not mask_i.any():
            continue
        if use_polar:
            last_pos = np.array(meta["last_position_canonical"], dtype=np.float32)
            last_heading = float(meta["last_heading"])
            canon_pred = reconstruct_trajectory_from_polar(
                predictions[i][mask_i], last_pos, last_heading
            )
            predictions_canonical[i, : canon_pred.shape[0]] = 0.0
            predictions_canonical[i, : canon_pred.shape[0]] = canon_pred
            canon_target = np.asarray(
                meta["decoder_target_cartesian"], dtype=np.float32
            )[: canon_pred.shape[0]]
            targets_canonical[i, : canon_target.shape[0]] = 0.0
            targets_canonical[i, : canon_target.shape[0]] = canon_target
        else:
            predictions_canonical[i] = predictions[i]
            targets_canonical[i] = targets[i]

    # Apply inverse transform to get back to original coordinates (yards)
    print(f"\nApplying inverse transform to get physical coordinates...")
    predictions_physical = np.zeros_like(predictions_canonical)
    targets_physical = np.zeros_like(targets_canonical)

    for i in range(len(predictions)):
        transform = all_transforms[i].get("transform")
        if transform is not None:
            predictions_physical[i] = transform.inverse_points(predictions_canonical[i])
            targets_physical[i] = transform.inverse_points(targets_canonical[i])
        else:
            # No transform (shouldn't happen with normalize=True)
            predictions_physical[i] = predictions_canonical[i]
            targets_physical[i] = targets_canonical[i]

    print(f"✓ Converted to physical coordinates (yards)")

    # Compute metrics
    print(f"\n{'='*60}")
    print("Results (Physical Coordinates - Yards)")
    print(f"{'='*60}")

    # Overall RMSE (only on valid frames) - using physical coordinates
    valid_predictions = predictions_physical[masks]
    valid_targets = targets_physical[masks]

    mse = np.mean((valid_predictions - valid_targets) ** 2)
    rmse = np.sqrt(mse)

    print(f"\nOverall Metrics:")
    print(f"  RMSE: {rmse:.4f} yards")
    print(f"  MSE: {mse:.4f} yards²")

    # Per-coordinate RMSE
    mse_x = np.mean((valid_predictions[:, 0] - valid_targets[:, 0]) ** 2)
    mse_y = np.mean((valid_predictions[:, 1] - valid_targets[:, 1]) ** 2)
    rmse_x = np.sqrt(mse_x)
    rmse_y = np.sqrt(mse_y)

    print(f"\nPer-Coordinate RMSE:")
    print(f"  X (field length): {rmse_x:.4f} yards")
    print(f"  Y (field width): {rmse_y:.4f} yards")

    # Per-frame RMSE (to see error accumulation)
    max_frames = masks.sum(axis=1).max()
    per_frame_rmse = []

    for t in range(int(max_frames)):
        frame_mask = masks[:, t]
        if frame_mask.sum() > 0:
            frame_pred = predictions_physical[frame_mask, t, :]
            frame_target = targets_physical[frame_mask, t, :]
            frame_mse = np.mean((frame_pred - frame_target) ** 2)
            frame_rmse = np.sqrt(frame_mse)
            per_frame_rmse.append(frame_rmse)

    print(f"\nPer-Frame RMSE (first 10 frames):")
    for i, fr_rmse in enumerate(per_frame_rmse[:10], 1):
        print(f"  Frame {i}: {fr_rmse:.4f} yards")

    if len(per_frame_rmse) > 10:
        print(f"  ...")
        print(f"  Frame {len(per_frame_rmse)}: {per_frame_rmse[-1]:.4f} yards")

    # MAE
    mae = np.mean(np.abs(valid_predictions - valid_targets))
    print(f"\nMean Absolute Error (MAE): {mae:.4f} yards")

    # Euclidean distance
    distances = np.sqrt(np.sum((valid_predictions - valid_targets) ** 2, axis=1))
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    p95_distance = np.percentile(distances, 95)
    max_distance = np.max(distances)

    print(f"\nEuclidean Distance Statistics:")
    print(f"  Mean: {mean_distance:.4f} yards")
    print(f"  Median: {median_distance:.4f} yards")
    print(f"  95th percentile: {p95_distance:.4f} yards")
    print(f"  Max: {max_distance:.4f} yards")

    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"{'='*60}\n")

    return {
        "rmse": rmse,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "mae": mae,
        "mean_distance": mean_distance,
        "median_distance": median_distance,
        "p95_distance": p95_distance,
        "per_frame_rmse": per_frame_rmse,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model checkpoint")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="fold1",
        choices=["fold1", "fold2"],
        help="Which fold to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    args = parser.parse_args()

    evaluate_model(args.checkpoint_path, args.fold, args.batch_size)
