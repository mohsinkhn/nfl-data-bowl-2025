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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as ptl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.configs.config_approach1_v0 import DataConfig, ModelConfig, TrainingConfig
from src.data import NFLTrajectoryDataset, TargetReceiverTrajectoryDataset, collate_fn
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
        test_target_df: Test target DataFrame (contains frame structure)
        data_config: Data configuration from trained model

    Returns:
        NFLTrajectoryDataset for test set
    """
    # For test set, we need to create dummy x, y values in the output
    # since the dataset expects them for initialization
    # We'll use zeros as placeholders - they won't be used for inference
    if "x" not in test_target_df.columns or "y" not in test_target_df.columns:
        test_target_df = test_target_df.copy()
        test_target_df["x"] = 0.0
        test_target_df["y"] = 0.0

    # Create temporary parquet files
    input_parquet, output_parquet = create_test_parquet_files(
        test_input_df, test_target_df
    )

    # Create dataset with same config as training
    dataset_type = getattr(data_config, "dataset_type", "baseline")
    dataset_cls = (
        TargetReceiverTrajectoryDataset
        if dataset_type == "target_receiver"
        else NFLTrajectoryDataset
    )

    target_kwargs: Dict[str, Any] = {}
    if dataset_cls is TargetReceiverTrajectoryDataset:
        target_kwargs["min_decoder_len"] = getattr(
            data_config, "target_min_decoder_len", 1
        )
        target_kwargs["filter_target_only"] = getattr(
            data_config, "target_only", True
        )

    test_dataset = dataset_cls(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        max_encoder_len=data_config.max_encoder_len,
        max_decoder_len=data_config.max_decoder_len,
        feature_cols=data_config.feature_cols,
        normalize=data_config.normalize,
        rotation_normalize=data_config.rotation_normalize,
        align_heading=getattr(data_config, "align_heading", True),
        use_player_role=getattr(data_config, "use_player_role", True),
        use_player_attributes=getattr(data_config, "use_player_attributes", True),
        use_polar_targets=getattr(data_config, "use_polar_targets", False),
        use_ball_residual_targets=getattr(
            data_config, "use_ball_residual_targets", False
        ),
        **target_kwargs,
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = ModelConfig(**checkpoint["hyper_parameters"]["model_config"])
    training_config = TrainingConfig(**checkpoint["hyper_parameters"]["training_config"])
    data_config = DataConfig(**checkpoint["hyper_parameters"]["data_config"])

    model_config.input_dim = test_dataset.input_dim
    model_config.output_dim = test_dataset.output_dim

    model = TrajectoryPredictionModule.load_from_checkpoint(
        checkpoint_path,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
    )
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
    use_polar = getattr(data_config, "use_polar_targets", False)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            # Move batch to device
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_mask = batch["decoder_mask"]
            encoder_mask = batch.get("encoder_mask")
            if encoder_mask is not None:
                encoder_mask = encoder_mask.to(device)
            metadata = batch["metadata"]

            # Get max decoder length for this batch
            max_len = decoder_mask.sum(dim=1).max().item()

            prediction_mode = getattr(model, "prediction_mode", "autoregressive")
            if prediction_mode == "autoregressive":
                if use_polar:
                    decoder_start = torch.zeros_like(decoder_input[:, 0, :])
                else:
                    decoder_start = decoder_input[:, 0, :]  # (batch, 2)
                batch_preds = model.model.predict(
                    encoder_inputs=encoder_input,
                    decoder_start=decoder_start,
                    prediction_steps=int(max_len),
                )
            else:
                batch_preds = model.model.predict(
                    encoder_inputs=encoder_input,
                    encoder_mask=encoder_mask,
                )
                if batch_preds.size(1) > max_len:
                    batch_preds = batch_preds[:, :max_len]

            # Move predictions back to CPU
            batch_preds = batch_preds.cpu().numpy()  # (batch, max_len, 2)

            # Process each sample in batch
            for i, meta in enumerate(metadata):
                pred = batch_preds[i]  # (max_len, 2)
                mask = decoder_mask[i].numpy()  # (max_len,)

                # Get valid predictions (only up to actual sequence length)
                valid_len = int(mask.sum())
                if use_polar:
                    last_pos = np.array(meta["last_position_canonical"], dtype=np.float32)
                    last_heading = float(meta["last_heading"])
                    pred_valid = reconstruct_trajectory_from_polar(
                        pred[:valid_len], last_pos, last_heading
                    )
                elif bool(meta.get("use_ball_residual_targets", False)):
                    landing = np.array(meta["ball_land_canonical"], dtype=np.float32)
                    pred_valid = landing - pred[:valid_len]
                else:
                    pred_valid = pred[:valid_len]

                transform = meta.get("transform")

                if transform is not None:
                    pred_denorm = transform.inverse_points(pred_valid)
                else:
                    pred_denorm = pred_valid

                # Store predictions with metadata
                predictions.append(
                    {
                        "game_id": meta["game_id"],
                        "play_id": meta["play_id"],
                        "nfl_id": meta["nfl_id"],
                        "predictions": pred_denorm,  # (valid_len, 2) in original coordinates
                        "num_frames": valid_len,
                        "is_target": bool(meta.get("is_target_receiver", False)),
                    }
                )

    print(f"\n✓ Inference complete")
    print(f"  Generated predictions for {len(predictions):,} samples")

    return predictions


def create_submission_file(
    predictions: List[Dict],
    test_target_df: pd.DataFrame,
    output_path: str,
    test_input_df: pd.DataFrame,
    fallback_strategy: str = "hold",
) -> None:
    """Format predictions as submission file with full-player coverage.

    When using the target-only model, defenders/off-ball receivers will not
    produce predictions. This helper fills their trajectories with a simple
    fallback (holding the last observed pre-throw position) so dashboards can
    still render every player.
    """
    print(f"\n{'='*60}")
    print("Creating Submission File")
    print(f"{'='*60}")

    # Structure frames and prediction lookups.
    frame_lookup: Dict[Tuple[int, int, int], List[int]] = {}
    for (game_id, play_id, nfl_id), group in test_target_df.groupby(
        ["game_id", "play_id", "nfl_id"]
    ):
        frame_ids = sorted(group["frame_id"].astype(int).tolist())
        frame_lookup[(int(game_id), int(play_id), int(nfl_id))] = frame_ids

    prediction_lookup: Dict[Tuple[int, int, int], Dict] = {
        (entry["game_id"], entry["play_id"], entry["nfl_id"]): entry
        for entry in predictions
    }

    # Cache last pre-throw position per player for fallback.
    last_positions = (
        test_input_df.sort_values(
            ["game_id", "play_id", "nfl_id", "frame_id"]
        )
        .groupby(["game_id", "play_id", "nfl_id"])[["x", "y"]]
        .last()
        .reset_index()
    )
    last_position_lookup: Dict[Tuple[int, int, int], Tuple[float, float]] = {
        (int(row.game_id), int(row.play_id), int(row.nfl_id)): (
            float(row.x),
            float(row.y),
        )
        for row in last_positions.itertuples(index=False)
    }

    fallback_strategy = fallback_strategy.lower()
    fallback_count = 0

    submission_rows: List[Dict[str, float]] = []

    for key, frame_ids in tqdm(frame_lookup.items(), desc="Building submission"):
        pred_entry = prediction_lookup.get(key)
        frames = frame_ids

        if pred_entry is not None:
            pred_coords = pred_entry["predictions"]
            if len(pred_coords) != len(frames):
                print(
                    f"Warning: Length mismatch for {key}: "
                    f"predictions={len(pred_coords)}, frames={len(frames)}"
                )
                n = min(len(pred_coords), len(frames))
                pred_coords = pred_coords[:n]
                frames = frames[:n]
        else:
            fallback_count += 1
            last_xy = last_position_lookup.get(key, (0.0, 0.0))
            if fallback_strategy == "hold" or len(frames) == 0:
                pred_coords = np.repeat(
                    np.array(last_xy, dtype=np.float32)[None, :], len(frames), axis=0
                )
            else:
                pred_coords = np.repeat(
                    np.array(last_xy, dtype=np.float32)[None, :], len(frames), axis=0
                )

        game_id, play_id, nfl_id = key
        for frame_id, (x, y) in zip(frames, pred_coords):
            row_id = f"{game_id}_{play_id}_{nfl_id}_{frame_id}"
            submission_rows.append({"id": row_id, "x": float(x), "y": float(y)})

    submission_df = pd.DataFrame(submission_rows)
    submission_df = submission_df.sort_values("id").reset_index(drop=True)
    submission_df.to_csv(output_path, index=False)

    print(f"\n✓ Submission file created")
    print(f"  Path: {output_path}")
    print(f"  Total predictions: {len(submission_df):,}")
    print(f"  Unique players: {len(frame_lookup):,}")
    if fallback_count:
        print(f"  Fallback trajectories applied: {fallback_count:,}")

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
    parser.add_argument(
        "--fallback_strategy",
        type=str,
        default="hold",
        choices=["hold"],
        help="Fallback trajectory policy for non-target players",
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
    create_submission_file(
        predictions,
        test_target_df,
        args.output_path,
        test_input_df=test_input_df,
        fallback_strategy=args.fallback_strategy,
    )

    print(f"\n{'='*60}")
    print("✓ Inference pipeline complete!")
    print(f"{'='*60}")
    print(f"\nSubmission saved to: {args.output_path}")
    print(f"Ready for upload to competition platform.")


if __name__ == "__main__":
    main()
