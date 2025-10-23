"""Evaluate model on validation set with proper inverse transformation.

This script:
1. Loads validation data
2. Runs inference with the trained model
3. Inverse transforms both predictions and ground truth
4. Calculates RMSE on the original coordinate space
5. Generates prediction CSV for dashboard
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from src.configs.config_approach1_v0 import DataConfig
from src.data import NFLTrajectoryDataset, collate_fn
from src.trainer import TrajectoryPredictionModule
from torch.utils.data import DataLoader


def evaluate_on_validation(
    checkpoint_path: str,
    val_input_parquet: str,
    val_output_parquet: str,
    output_predictions_path: str = None,
    batch_size: int = 32,
):
    """Evaluate model on validation set with inverse transformation.

    Args:
        checkpoint_path: Path to trained model checkpoint
        val_input_parquet: Path to validation input parquet
        val_output_parquet: Path to validation output parquet
        output_predictions_path: Optional path to save predictions CSV for dashboard
        batch_size: Batch size for inference
    """
    print(f"\n{'='*60}")
    print(f"Validation Set Evaluation (Inverse Transformed)")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Val input: {val_input_parquet}")
    print(f"Val output: {val_output_parquet}")

    # Load checkpoint to get configs
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    data_config = DataConfig(**checkpoint["hyper_parameters"]["data_config"])
    print(f"✓ Configs loaded")

    # Create validation dataset
    print(f"\nCreating validation dataset...")
    val_dataset = NFLTrajectoryDataset(
        input_parquet=val_input_parquet,
        output_parquet=val_output_parquet,
        max_encoder_len=data_config.max_encoder_len,
        max_decoder_len=data_config.max_decoder_len,
        feature_cols=data_config.feature_cols,
        normalize=data_config.normalize,
        rotation_normalize=data_config.rotation_normalize,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    print(f"✓ Validation dataset: {len(val_dataset):,} samples")

    # Load model
    print(f"\nLoading model...")
    model = TrajectoryPredictionModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✓ Model loaded on {device}")

    # Run inference and collect results
    print(f"\nRunning inference on validation set...")

    all_predictions = []
    all_ground_truth = []
    all_squared_errors = []
    all_metadata = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_target = batch["decoder_target"]  # Keep on CPU
            decoder_mask = batch["decoder_mask"]
            metadata = batch["metadata"]

            # Get predictions
            decoder_start = decoder_input[:, 0, :]
            max_len = decoder_target.size(1)

            predictions = model.model.predict(
                encoder_inputs=encoder_input,
                decoder_start=decoder_start,
                prediction_steps=int(max_len),
            )

            # Convert to CPU
            predictions = predictions.cpu().numpy()  # (batch, max_len, 2)
            targets = decoder_target.numpy()  # (batch, max_len, 2)
            masks = decoder_mask.numpy()  # (batch, max_len)

            # Process each sample
            for i in range(len(metadata)):
                meta = metadata[i]
                pred = predictions[i]  # (max_len, 2)
                target = targets[i]  # (max_len, 2)
                mask = masks[i]  # (max_len,)

                # Get valid length
                valid_len = int(mask.sum())
                pred_valid = pred[:valid_len]
                target_valid = target[:valid_len]

                # Get transform for inverse normalization
                transform = meta.get("transform")

                if transform is not None:
                    # Inverse transform both prediction and ground truth
                    pred_denorm = transform.inverse_transform(pred_valid)
                    target_denorm = transform.inverse_transform(target_valid)
                else:
                    pred_denorm = pred_valid
                    target_denorm = target_valid

                # Calculate squared errors per point
                squared_errors = np.sum((pred_denorm - target_denorm) ** 2, axis=1)

                all_predictions.append(pred_denorm)
                all_ground_truth.append(target_denorm)
                all_squared_errors.extend(squared_errors.tolist())
                all_metadata.append(meta)

    # Calculate overall RMSE
    mse = np.mean(all_squared_errors)
    rmse = np.sqrt(mse)

    print(f"\n{'='*60}")
    print(f"Validation Results (Inverse Transformed)")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_predictions):,}")
    print(f"Total predictions: {len(all_squared_errors):,}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"{'='*60}")

    # Calculate per-sample RMSE statistics
    per_sample_rmse = []
    for i in range(len(all_predictions)):
        pred = all_predictions[i]
        target = all_ground_truth[i]
        sample_mse = np.mean(np.sum((pred - target) ** 2, axis=1))
        per_sample_rmse.append(np.sqrt(sample_mse))

    print(f"\nPer-Sample RMSE Statistics:")
    print(f"  Mean: {np.mean(per_sample_rmse):.6f}")
    print(f"  Std: {np.std(per_sample_rmse):.6f}")
    print(f"  Min: {np.min(per_sample_rmse):.6f}")
    print(f"  Max: {np.max(per_sample_rmse):.6f}")
    print(f"  Median: {np.median(per_sample_rmse):.6f}")
    print(f"  25th percentile: {np.percentile(per_sample_rmse, 25):.6f}")
    print(f"  75th percentile: {np.percentile(per_sample_rmse, 75):.6f}")

    # Save predictions for dashboard if requested
    if output_predictions_path:
        print(f"\nSaving predictions for dashboard...")

        # Load original output data to get frame_ids
        output_df = pd.read_parquet(val_output_parquet)

        prediction_rows = []

        for i, meta in enumerate(all_metadata):
            game_id = meta["game_id"]
            play_id = meta["play_id"]
            nfl_id = meta["nfl_id"]
            pred_coords = all_predictions[i]

            # Get frame_ids for this sample
            sample_output = output_df[
                (output_df["game_id"] == game_id)
                & (output_df["play_id"] == play_id)
                & (output_df["nfl_id"] == nfl_id)
            ].sort_values("frame_id")

            frame_ids = sample_output["frame_id"].values[: len(pred_coords)]

            # Create rows
            for j, frame_id in enumerate(frame_ids):
                prediction_rows.append(
                    {
                        "game_id": game_id,
                        "play_id": play_id,
                        "nfl_id": nfl_id,
                        "frame_id": frame_id,
                        "x": float(pred_coords[j, 0]),
                        "y": float(pred_coords[j, 1]),
                    }
                )

        # Save to CSV
        pred_df = pd.DataFrame(prediction_rows)
        pred_df = pred_df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
        pred_df.to_csv(output_predictions_path, index=False)

        print(f"✓ Predictions saved to: {output_predictions_path}")
        print(f"  Total rows: {len(pred_df):,}")
        print(f"  Unique plays: {pred_df.groupby(['game_id', 'play_id']).ngroups}")
        print(
            f"  Unique players: {pred_df.groupby(['game_id', 'play_id', 'nfl_id']).ngroups}"
        )

    return {
        "rmse": rmse,
        "mse": mse,
        "per_sample_rmse": per_sample_rmse,
        "predictions": all_predictions,
        "ground_truth": all_ground_truth,
        "metadata": all_metadata,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model on validation set with inverse transformation"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--val_input",
        type=str,
        default="data/processed/fold1/val_input.parquet",
        help="Path to validation input parquet",
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="data/processed/fold1/val_output.parquet",
        help="Path to validation output parquet",
    )
    parser.add_argument(
        "--output_predictions",
        type=str,
        default=None,
        help="Optional: path to save predictions CSV for dashboard",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )

    args = parser.parse_args()

    results = evaluate_on_validation(
        checkpoint_path=args.checkpoint_path,
        val_input_parquet=args.val_input,
        val_output_parquet=args.val_output,
        output_predictions_path=args.output_predictions,
        batch_size=args.batch_size,
    )

    print(f"\n✓ Evaluation complete!")
