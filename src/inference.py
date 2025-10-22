"""Inference pipeline for NFL trajectory prediction.

This module provides functions to:
1. Load test data
2. Create test datasets
3. Run inference with trained models
4. Generate submission files

The inference pipeline automatically loads model configs and transform params
from checkpoints for reproducible predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as ptl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.configs.config_approach1_v0 import DataConfig, ModelConfig, TrainingConfig
from src.data import NFLTrajectoryDataset, collate_fn
from src.trainer import TrajectoryPredictionModule


def load_test_data(
    test_input_path: str, test_target_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load test input and target structure from CSV files.

    Args:
        test_input_path: Path to test_input.csv
        test_target_path: Path to test.csv (contains target structure)

    Returns:
        Tuple of (test_input_df, test_target_df)
    """
    print(f"Loading test data...")
    print(f"  Input: {test_input_path}")
    print(f"  Target: {test_target_path}")

    test_input = pd.read_csv(test_input_path)
    test_target = pd.read_csv(test_target_path)

    print(f"\n✓ Test data loaded")
    print(f"  Input rows: {len(test_input):,}")
    print(f"  Target rows: {len(test_target):,}")
    print(
        f"  Unique plays (input): {test_input[['game_id', 'play_id', 'nfl_id']].drop_duplicates().shape[0]:,}"
    )
    print(
        f"  Unique plays (target): {test_target[['game_id', 'play_id', 'nfl_id']].drop_duplicates().shape[0]:,}"
    )

    return test_input, test_target


def create_test_parquet_files(
    test_input_df: pd.DataFrame,
    test_target_df: pd.DataFrame,
    output_dir: str = "data/processed/test",
) -> Tuple[str, str]:
    """Create temporary parquet files for test data.

    Args:
        test_input_df: Test input DataFrame
        test_target_df: Test target DataFrame
        output_dir: Directory to save parquet files

    Returns:
        Tuple of (input_parquet_path, output_parquet_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_parquet = str(output_path / "test_input.parquet")
    output_parquet = str(output_path / "test_output.parquet")

    print(f"\nCreating temporary parquet files...")
    test_input_df.to_parquet(input_parquet, index=False)
    test_target_df.to_parquet(output_parquet, index=False)
    print(f"  Saved to: {output_dir}")

    return input_parquet, output_parquet


def create_test_dataset(
    test_input_df: pd.DataFrame,
    test_target_df: pd.DataFrame,
    data_config: DataConfig,
) -> NFLTrajectoryDataset:
    """Create dataset for test set.

    Args:
        test_input_df: Test input DataFrame
        test_target_df: Test target DataFrame
        data_config: Data configuration from trained model

    Returns:
        NFLTrajectoryDataset for test set
    """
    # Create temporary parquet files
    input_parquet, output_parquet = create_test_parquet_files(
        test_input_df, test_target_df
    )

    # Create dataset with same config as training
    test_dataset = NFLTrajectoryDataset(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        max_encoder_len=data_config.max_encoder_len,
        max_decoder_len=data_config.max_decoder_len,
        feature_cols=data_config.feature_cols,
        normalize=data_config.normalize,
        rotation_normalize=data_config.rotation_normalize,
    )

    return test_dataset


def run_inference(
    checkpoint_path: str,
    test_dataset: NFLTrajectoryDataset,
    batch_size: int = 64,
    num_workers: int = 4,
) -> List[Dict]:
    """Run inference on test set and return predictions.

    Args:
        checkpoint_path: Path to trained model checkpoint
        test_dataset: Test dataset
        batch_size: Batch size for inference
        num_workers: Number of data loading workers

    Returns:
        List of prediction dictionaries with denormalized coordinates
    """
    print(f"\n{'='*60}")
    print("Running Inference")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Batch size: {batch_size}")

    # Load model from checkpoint (configs and transform_params restored automatically)
    print(f"\nLoading model from checkpoint...")
    model = TrajectoryPredictionModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()

    print(f"✓ Model loaded")
    print(f"  Encoder: {model.model_config.encoder_type}")
    print(f"  Decoder: {model.model_config.decoder_type}")
    print(f"  Hidden dim: {model.model_config.hidden_dim}")
    print(f"  Num layers: {model.model_config.num_layers}")

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Run predictions
    print(f"\nGenerating predictions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            # Move batch to device
            encoder_input = batch["encoder_input"].to(device)
            decoder_mask = batch["decoder_mask"]
            metadata = batch["metadata"]

            # Get max decoder length for this batch
            max_len = decoder_mask.sum(dim=1).max().item()

            # Run model prediction (no teacher forcing)
            batch_preds = model.model.predict(encoder_input, max_len=int(max_len))

            # Move predictions back to CPU
            batch_preds = batch_preds.cpu().numpy()  # (batch, max_len, 2)

            # Process each sample in batch
            for i in range(len(metadata)):
                meta = metadata[i]
                pred = batch_preds[i]  # (max_len, 2)
                mask = decoder_mask[i].numpy()  # (max_len,)

                # Get valid predictions (only up to actual sequence length)
                valid_len = int(mask.sum())
                pred_valid = pred[:valid_len]  # (valid_len, 2)

                # Get transform for this sample from dataset
                sample_idx = None
                for idx, key in enumerate(test_dataset.samples):
                    if (
                        key[0] == meta["game_id"]
                        and key[1] == meta["play_id"]
                        and key[2] == meta["nfl_id"]
                    ):
                        sample_idx = idx
                        break

                if sample_idx is None:
                    print(
                        f"Warning: Could not find sample for {meta['game_id']}, {meta['play_id']}, {meta['nfl_id']}"
                    )
                    continue

                # Get the transform for denormalization
                sample = test_dataset[sample_idx]
                transform = sample["metadata"]["transform"]

                # Denormalize predictions
                pred_denorm = transform.inverse_transform(pred_valid)

                # Store predictions with metadata
                predictions.append(
                    {
                        "game_id": meta["game_id"],
                        "play_id": meta["play_id"],
                        "nfl_id": meta["nfl_id"],
                        "predictions": pred_denorm,  # (valid_len, 2) in original coordinates
                        "num_frames": valid_len,
                    }
                )

    print(f"\n✓ Inference complete")
    print(f"  Generated predictions for {len(predictions):,} samples")

    return predictions


def create_submission_file(
    predictions: List[Dict],
    test_target_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Format predictions as submission file.

    Creates a CSV file with columns: id, x, y
    where id format is: {game_id}_{play_id}_{nfl_id}_{frame_id}

    Args:
        predictions: List of prediction dictionaries from run_inference
        test_target_df: Test target DataFrame (contains frame_id structure)
        output_path: Path to save submission CSV
    """
    print(f"\n{'='*60}")
    print("Creating Submission File")
    print(f"{'='*60}")

    # Create a lookup for frame_ids from test_target
    frame_lookup = {}
    for (game_id, play_id, nfl_id), group in test_target_df.groupby(
        ["game_id", "play_id", "nfl_id"]
    ):
        frame_ids = sorted(group["frame_id"].values)
        frame_lookup[(game_id, play_id, nfl_id)] = frame_ids

    # Build submission rows
    submission_rows = []

    for pred_dict in tqdm(predictions, desc="Building submission"):
        game_id = pred_dict["game_id"]
        play_id = pred_dict["play_id"]
        nfl_id = pred_dict["nfl_id"]
        pred_coords = pred_dict["predictions"]  # (num_frames, 2)

        # Get frame_ids for this player
        key = (game_id, play_id, nfl_id)
        if key not in frame_lookup:
            print(f"Warning: No frame_ids found for {key}")
            continue

        frame_ids = frame_lookup[key]

        # Check length match
        if len(pred_coords) != len(frame_ids):
            print(
                f"Warning: Length mismatch for {key}: "
                f"predictions={len(pred_coords)}, frames={len(frame_ids)}"
            )
            # Use minimum length
            n = min(len(pred_coords), len(frame_ids))
            pred_coords = pred_coords[:n]
            frame_ids = frame_ids[:n]

        # Create submission rows
        for frame_id, (x, y) in zip(frame_ids, pred_coords):
            row_id = f"{game_id}_{play_id}_{nfl_id}_{frame_id}"
            submission_rows.append({"id": row_id, "x": float(x), "y": float(y)})

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Sort by id for consistency
    submission_df = submission_df.sort_values("id").reset_index(drop=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)

    print(f"\n✓ Submission file created")
    print(f"  Path: {output_path}")
    print(f"  Total predictions: {len(submission_df):,}")
    print(f"  Unique players: {len(predictions):,}")

    # Show sample
    print(f"\nSample predictions:")
    print(submission_df.head(10).to_string(index=False))


def main() -> None:
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(
        description="Run inference on test set for NFL trajectory prediction"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--test_input",
        type=str,
        default="data/test_input.csv",
        help="Path to test input CSV",
    )
    parser.add_argument(
        "--test_target",
        type=str,
        default="data/test.csv",
        help="Path to test target CSV (contains frame structure)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="submission.csv",
        help="Path to save submission CSV",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("NFL Trajectory Prediction - Inference")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Test input: {args.test_input}")
    print(f"  Test target: {args.test_target}")
    print(f"  Output: {args.output_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {args.num_workers}")

    # Load checkpoint to extract configs
    print(f"\nLoading checkpoint to extract configs...")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    # Extract configs from checkpoint
    data_config = DataConfig(**checkpoint["hyper_parameters"]["data_config"])
    model_config = ModelConfig(**checkpoint["hyper_parameters"]["model_config"])

    print(f"✓ Configs loaded from checkpoint")
    print(f"  Max encoder length: {data_config.max_encoder_len}")
    print(f"  Max decoder length: {data_config.max_decoder_len}")
    print(f"  Features: {data_config.feature_cols}")

    # Load test data
    test_input_df, test_target_df = load_test_data(args.test_input, args.test_target)

    # Create test dataset (configs loaded from checkpoint)
    test_dataset = create_test_dataset(test_input_df, test_target_df, data_config)

    # Run inference (model loads configs and transform_params internally)
    predictions = run_inference(
        args.checkpoint_path,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create submission file
    create_submission_file(predictions, test_target_df, args.output_path)

    print(f"\n{'='*60}")
    print("✓ Inference pipeline complete!")
    print(f"{'='*60}")
    print(f"\nSubmission saved to: {args.output_path}")
    print(f"Ready for upload to competition platform.")


if __name__ == "__main__":
    main()
