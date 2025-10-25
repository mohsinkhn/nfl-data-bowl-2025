"""Prepare train/validation splits from raw CSV files.

This script loads input/output CSV files and creates train/val splits.
No statistics computation - keep it focused on data loading and splitting.

Fold strategies:
- fold1: train=weeks 1-14, val=weeks 15-18 (temporal validation)
- fold2: train=weeks 1-8,13-18, val=weeks 9-12 (mid-season validation)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_fold_weeks(fold: str) -> tuple[list[int], list[int]]:
    """Get train and validation week numbers for specified fold."""
    if fold == "fold1":
        return list(range(1, 15)), list(range(15, 19))
    elif fold == "fold2":
        return list(range(1, 9)) + list(range(13, 19)), list(range(9, 13))
    else:
        raise ValueError(f"Invalid fold: {fold}. Must be 'fold1' or 'fold2'.")


def load_weeks(data_dir: Path, weeks: list[int], file_type: str) -> pd.DataFrame:
    """Load and concatenate CSV files for specified weeks."""
    dfs = []
    for week in tqdm(weeks, desc=f"Loading {file_type}"):
        file_path = data_dir / f"{file_type}_2023_w{week:02d}.csv"
        df = pd.read_csv(file_path)
        df["week"] = week
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def prepare_fold(data_dir: Path, output_dir: Path, fold: str) -> None:
    """Prepare train/val data for specified fold."""
    print(f"\n{'='*60}")
    print(f"Preparing {fold}")
    print(f"{'='*60}")

    train_weeks, val_weeks = get_fold_weeks(fold)
    print(f"Train weeks: {train_weeks}")
    print(f"Val weeks: {val_weeks}\n")

    fold_dir = output_dir / fold
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    print("Loading training data...")
    train_input = load_weeks(data_dir, train_weeks, "input")
    train_output = load_weeks(data_dir, train_weeks, "output")

    # Load validation data
    print("\nLoading validation data...")
    val_input = load_weeks(data_dir, val_weeks, "input")
    val_output = load_weeks(data_dir, val_weeks, "output")

    def add_ball_features(df: pd.DataFrame) -> pd.DataFrame:
        delta_x = df["ball_land_x"] - df["x"]
        delta_y = df["ball_land_y"] - df["y"]
        df["ball_angle"] = np.degrees(np.arctan2(delta_y, delta_x)) % 360.0
        df["dir"] = (90.0 - df["dir"]) % 360.0
        df["o"] = (90.0 - df["o"]) % 360.0
        return df

    train_input = add_ball_features(train_input)
    val_input = add_ball_features(val_input)

    # Save to parquet
    print("\nSaving files...")
    train_input.to_parquet(fold_dir / "train_input.parquet", index=False)
    train_output.to_parquet(fold_dir / "train_output.parquet", index=False)
    val_input.to_parquet(fold_dir / "val_input.parquet", index=False)
    val_output.to_parquet(fold_dir / "val_output.parquet", index=False)

    print(f"\n✓ {fold} complete: {fold_dir}")
    print(
        f"  Train: {len(train_input):,} input rows, {len(train_output):,} output rows"
    )
    print(f"  Val: {len(val_input):,} input rows, {len(val_output):,} output rows")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare NFL trajectory prediction data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/train",
        help="Path to raw training data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Path to save processed data",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="both",
        choices=["fold1", "fold2", "both"],
        help="Which fold(s) to prepare",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.fold in ["fold1", "both"]:
        prepare_fold(data_dir, output_dir, "fold1")

    if args.fold in ["fold2", "both"]:
        prepare_fold(data_dir, output_dir, "fold2")

    print(f"\n{'='*60}")
    print("✓ Data preparation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
