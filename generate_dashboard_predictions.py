"""Generate predictions for dashboard visualization.

This script generates predictions for a specific week's data that can be
loaded into the Streamlit dashboard for visualization.
"""

import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from src.configs.config_approach1_v0 import DataConfig
from src.data import NFLTrajectoryDataset, collate_fn
from src.trainer import TrajectoryPredictionModule
from torch.utils.data import DataLoader


def generate_predictions_for_week(
    checkpoint_path: str,
    week: int,
    output_path: str,
    batch_size: int = 32,
):
    """Generate predictions for all plays in a given week.

    Args:
        checkpoint_path: Path to trained model checkpoint
        week: Week number (1-18)
        output_path: Path to save predictions CSV
        batch_size: Batch size for inference
    """
    print(f"\n{'='*60}")
    print(f"Generating Predictions for Week {week}")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")

    # Load checkpoint to get configs
    print(f"\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    data_config = DataConfig(**checkpoint["hyper_parameters"]["data_config"])

    print(f"✓ Configs loaded")

    # Load week data
    print(f"\nLoading week {week} data...")
    input_file = f"data/train/input_2023_w{week:02d}.csv"
    output_file = f"data/train/output_2023_w{week:02d}.csv"

    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(output_file)

    print(
        f"✓ Data loaded: {len(input_df):,} input rows, {len(output_df):,} output rows"
    )

    # Create temporary parquet files
    temp_dir = Path("data/processed/temp_dashboard")
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_parquet = str(temp_dir / "input.parquet")
    output_parquet = str(temp_dir / "output.parquet")

    input_df.to_parquet(input_parquet, index=False)
    output_df.to_parquet(output_parquet, index=False)

    # Create dataset
    print(f"\nCreating dataset...")
    dataset = NFLTrajectoryDataset(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        max_encoder_len=data_config.max_encoder_len,
        max_decoder_len=data_config.max_decoder_len,
        feature_cols=data_config.feature_cols,
        normalize=data_config.normalize,
        rotation_normalize=data_config.rotation_normalize,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Load model
    print(f"\nLoading model...")
    model = TrajectoryPredictionModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✓ Model loaded on {device}")

    # Generate predictions
    print(f"\nGenerating predictions for {len(dataset):,} samples...")

    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            decoder_mask = batch["decoder_mask"]
            metadata = batch["metadata"]

            # Get predictions
            decoder_start = decoder_input[:, 0, :]
            max_len = data_config.max_decoder_len

            predictions = model.model.predict(
                encoder_inputs=encoder_input,
                decoder_start=decoder_start,
                prediction_steps=int(max_len),
            )

            # Convert to CPU
            predictions = predictions.cpu().numpy()

            # Process each sample in batch
            for i in range(len(metadata)):
                meta = metadata[i]
                pred = predictions[i]
                mask = decoder_mask[i].numpy()

                # Get valid predictions
                valid_len = int(mask.sum())
                pred_valid = pred[:valid_len]

                # Get transform for denormalization
                transform = meta.get("transform")
                if transform is not None:
                    # Denormalize to physical coordinates
                    pred_denorm = transform.inverse_transform(pred_valid)
                else:
                    pred_denorm = pred_valid

                # Get corresponding frame_ids from output_df
                game_id = meta["game_id"]
                play_id = meta["play_id"]
                nfl_id = meta["nfl_id"]

                output_frames = output_df[
                    (output_df["game_id"] == game_id)
                    & (output_df["play_id"] == play_id)
                    & (output_df["nfl_id"] == nfl_id)
                ].sort_values("frame_id")

                frame_ids = output_frames["frame_id"].values[:valid_len]

                # Create prediction rows
                for j, frame_id in enumerate(frame_ids):
                    all_predictions.append(
                        {
                            "game_id": game_id,
                            "play_id": play_id,
                            "nfl_id": nfl_id,
                            "frame_id": frame_id,
                            "x": float(pred_denorm[j, 0]),
                            "y": float(pred_denorm[j, 1]),
                        }
                    )

    # Save predictions
    pred_df = pd.DataFrame(all_predictions)
    pred_df = pred_df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
    pred_df.to_csv(output_path, index=False)

    print(f"\n✓ Predictions saved: {len(pred_df):,} rows")
    print(f"  Unique games: {pred_df['game_id'].nunique()}")
    print(f"  Unique plays: {pred_df.groupby(['game_id', 'play_id']).ngroups}")
    print(
        f"  Unique players: {pred_df.groupby(['game_id', 'play_id', 'nfl_id']).ngroups}"
    )

    print(f"\n{'='*60}")
    print("✓ Complete! Load this file in the dashboard.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate predictions for dashboard visualization"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        choices=range(1, 19),
        help="Week number (1-18)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save predictions (default: predictions_week_{week}.csv)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"predictions_week_{args.week:02d}.csv"

    generate_predictions_for_week(
        args.checkpoint_path,
        args.week,
        args.output_path,
        args.batch_size,
    )
