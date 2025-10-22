"""Visualize normalized trajectories from prepared data.

Loads parquet files, applies coordinate normalization, and plots
the transformed trajectories to verify data preparation.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.normalization import CoordinateTransform


def load_sample_plays(fold_dir: Path, split: str, n_plays: int = 6):
    """Load sample plays from parquet files."""
    input_df = pd.read_parquet(fold_dir / f"{split}_input.parquet")
    output_df = pd.read_parquet(fold_dir / f"{split}_output.parquet")

    # Get unique plays and sample randomly (no fixed seed for variety)
    unique_plays = input_df[["game_id", "play_id"]].drop_duplicates()
    sample_plays = unique_plays.sample(n=min(n_plays, len(unique_plays)))

    return input_df, output_df, sample_plays


def normalize_play(
    input_df: pd.DataFrame, output_df: pd.DataFrame, game_id: int, play_id: int
):
    """Apply normalization to a single play and test reverse transformation."""
    # Filter for this play
    play_input = input_df[
        (input_df["game_id"] == game_id) & (input_df["play_id"] == play_id)
    ].copy()
    play_output = output_df[
        (output_df["game_id"] == game_id) & (output_df["play_id"] == play_id)
    ].copy()

    if len(play_input) == 0:
        return None, None, None

    # Get play direction
    play_direction = play_input["play_direction"].iloc[0]

    # Store transformers for testing inverse
    transformers = {}
    max_reconstruction_error = 0.0

    # Normalize each player separately
    for nfl_id in play_input["nfl_id"].unique():
        # Get player data
        player_input = play_input[play_input["nfl_id"] == nfl_id].sort_values(
            "frame_id"
        )
        player_output = play_output[play_output["nfl_id"] == nfl_id].sort_values(
            "frame_id"
        )

        if len(player_input) == 0:
            continue

        # Create transformer and fit on player's input trajectory
        transformer = CoordinateTransform(
            field_dims=(120.0, 53.3), normalize_field=True, normalize_rotation=True
        )

        # Prepare data for fitting (needs x, y, and other features)
        player_data = player_input[["x", "y", "s", "a", "dir", "o"]].values
        transformer.fit(player_data, play_direction)
        transformers[nfl_id] = transformer

        # Transform input coordinates and angles
        coords_in = player_input[["x", "y"]].values
        coords_full = np.column_stack([coords_in, np.zeros((len(coords_in), 4))])
        coords_norm = transformer.transform(coords_full)

        play_input.loc[play_input["nfl_id"] == nfl_id, "x_norm"] = coords_norm[:, 0]
        play_input.loc[play_input["nfl_id"] == nfl_id, "y_norm"] = coords_norm[:, 1]

        # Transform direction and orientation angles
        angles = player_input[["dir", "o"]].values
        if transformer.normalize_rotation and transformer.angle is not None:
            angles_norm = (angles + np.degrees(transformer.angle)) % 360
        else:
            angles_norm = angles

        play_input.loc[play_input["nfl_id"] == nfl_id, "dir_norm"] = angles_norm[:, 0]
        play_input.loc[play_input["nfl_id"] == nfl_id, "o_norm"] = angles_norm[:, 1]

        # Test inverse transformation on input
        coords_reconstructed = transformer.inverse_transform(coords_norm)
        reconstruction_error = np.mean(np.abs(coords_reconstructed[:, :2] - coords_in))
        max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)

        play_input.loc[play_input["nfl_id"] == nfl_id, "x_reconstructed"] = (
            coords_reconstructed[:, 0]
        )
        play_input.loc[play_input["nfl_id"] == nfl_id, "y_reconstructed"] = (
            coords_reconstructed[:, 1]
        )

        # Transform output coordinates if they exist
        if len(player_output) > 0:
            coords_out = player_output[["x", "y"]].values
            coords_out_full = np.column_stack(
                [coords_out, np.zeros((len(coords_out), 4))]
            )
            coords_out_norm = transformer.transform(coords_out_full)

            play_output.loc[play_output["nfl_id"] == nfl_id, "x_norm"] = (
                coords_out_norm[:, 0]
            )
            play_output.loc[play_output["nfl_id"] == nfl_id, "y_norm"] = (
                coords_out_norm[:, 1]
            )

    return play_input, play_output, max_reconstruction_error


def plot_comparison_trajectory(
    play_input: pd.DataFrame,
    play_output: pd.DataFrame,
    game_id: int,
    play_id: int,
    reconstruction_error: float,
    output_path: Path,
):
    """Plot side-by-side comparison of original and normalized trajectories."""

    # Get play direction for title
    play_direction = play_input["play_direction"].iloc[0]

    # Filter to only players with output trajectories
    players_with_output = []
    for nfl_id in play_input["nfl_id"].unique():
        player_output = play_output[play_output["nfl_id"] == nfl_id]
        if len(player_output) > 0:
            players_with_output.append(nfl_id)

    if len(players_with_output) == 0:
        print(f"  ⚠ Skipping - no players with output trajectories")
        return

    n_players = len(players_with_output)

    # Create figure: 1 plot for original, n_players plots for normalized
    fig = plt.figure(figsize=(22, 6 * ((n_players + 1) // 2)))

    # Calculate grid layout: original on left, normalized players on right in grid
    n_rows = max(1, (n_players + 1) // 2)

    # Original plot takes full left column
    ax_orig = plt.subplot2grid((n_rows, 4), (0, 0), rowspan=n_rows, colspan=2)

    # Get color palette
    colors = plt.cm.tab20(np.linspace(0, 1, len(players_with_output)))

    # Arrow scale factor
    arrow_scale = 2.0

    # Plot original coordinates (all players together)
    for idx, nfl_id in enumerate(players_with_output):
        player_input = play_input[play_input["nfl_id"] == nfl_id].sort_values(
            "frame_id"
        )
        player_output = play_output[play_output["nfl_id"] == nfl_id].sort_values(
            "frame_id"
        )

        color = colors[idx]
        is_target = player_input["player_to_predict"].iloc[0] == 1
        alpha = 0.9 if is_target else 0.5
        linewidth = 2.5 if is_target else 1.5

        # Plot input trajectory
        ax_orig.plot(
            player_input["x"],
            player_input["y"],
            "o-",
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            markersize=5,
            label=f"Player {nfl_id}" + (" [TARGET]" if is_target else ""),
        )

        # Mark last input position
        last_x = player_input["x"].iloc[-1]
        last_y = player_input["y"].iloc[-1]
        ax_orig.plot(
            last_x,
            last_y,
            "o",
            color=color,
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=2,
            zorder=5,
        )

        # Draw direction and orientation arrows at last position
        last_dir = player_input["dir"].iloc[-1]
        last_o = player_input["o"].iloc[-1]

        # Direction arrow (blue)
        dx_dir = arrow_scale * np.cos(np.radians(last_dir))
        dy_dir = arrow_scale * np.sin(np.radians(last_dir))
        ax_orig.arrow(
            last_x,
            last_y,
            dx_dir,
            dy_dir,
            head_width=0.8,
            head_length=0.5,
            fc="blue",
            ec="blue",
            alpha=0.6,
            linewidth=1.5,
            zorder=4,
        )

        # Orientation arrow (green)
        dx_o = arrow_scale * np.cos(np.radians(last_o))
        dy_o = arrow_scale * np.sin(np.radians(last_o))
        ax_orig.arrow(
            last_x,
            last_y,
            dx_o,
            dy_o,
            head_width=0.8,
            head_length=0.5,
            fc="green",
            ec="green",
            alpha=0.6,
            linewidth=1.5,
            zorder=4,
        )

        # Plot output trajectory
        ax_orig.plot(
            player_output["x"],
            player_output["y"],
            "s--",
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            markersize=6,
        )

    # Add arrow legend
    ax_orig.arrow(
        0,
        0,
        0,
        0,
        head_width=0.8,
        fc="blue",
        ec="blue",
        alpha=0.6,
        label="Direction (dir)",
    )
    ax_orig.arrow(
        0,
        0,
        0,
        0,
        head_width=0.8,
        fc="green",
        ec="green",
        alpha=0.6,
        label="Orientation (o)",
    )

    ax_orig.set_xlabel("X (yards)", fontsize=12, fontweight="bold")
    ax_orig.set_ylabel("Y (yards)", fontsize=12, fontweight="bold")
    ax_orig.set_title(
        f"Original Coordinates\nDirection: {play_direction}",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax_orig.grid(True, alpha=0.3, linestyle="--")
    ax_orig.set_aspect("equal")
    ax_orig.legend(loc="best", fontsize=8, framealpha=0.9, edgecolor="black")

    # Get data bounds for consistent aspect ratio
    all_x = play_input["x"].values
    all_y = play_input["y"].values
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()

    # Plot normalized coordinates (one subplot per player)
    norm_arrow_scale = 0.05

    for idx, nfl_id in enumerate(players_with_output):
        # Calculate subplot position
        row = idx // 2
        col = 2 + (idx % 2)

        ax = plt.subplot2grid((n_rows, 4), (row, col), colspan=1)

        player_input = play_input[play_input["nfl_id"] == nfl_id].sort_values(
            "frame_id"
        )
        player_output = play_output[play_output["nfl_id"] == nfl_id].sort_values(
            "frame_id"
        )

        color = colors[idx]
        is_target = player_input["player_to_predict"].iloc[0] == 1

        # Plot input trajectory
        ax.plot(
            player_input["x_norm"],
            player_input["y_norm"],
            "o-",
            color=color,
            alpha=0.9,
            linewidth=2,
            markersize=5,
            label="Input",
        )

        # Mark last input position (should be at origin)
        last_x_norm = player_input["x_norm"].iloc[-1]
        last_y_norm = player_input["y_norm"].iloc[-1]
        ax.plot(
            last_x_norm,
            last_y_norm,
            "o",
            color=color,
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=2,
            zorder=5,
        )

        # Draw normalized direction and orientation arrows at last position
        if "dir_norm" in player_input.columns:
            last_dir_norm = player_input["dir_norm"].iloc[-1]
            last_o_norm = player_input["o_norm"].iloc[-1]

            # Direction arrow (blue)
            dx_dir_norm = norm_arrow_scale * np.cos(np.radians(last_dir_norm))
            dy_dir_norm = norm_arrow_scale * np.sin(np.radians(last_dir_norm))
            ax.arrow(
                last_x_norm,
                last_y_norm,
                dx_dir_norm,
                dy_dir_norm,
                head_width=0.015,
                head_length=0.01,
                fc="blue",
                ec="blue",
                alpha=0.7,
                linewidth=2,
                zorder=4,
            )

            # Orientation arrow (green)
            dx_o_norm = norm_arrow_scale * np.cos(np.radians(last_o_norm))
            dy_o_norm = norm_arrow_scale * np.sin(np.radians(last_o_norm))
            ax.arrow(
                last_x_norm,
                last_y_norm,
                dx_o_norm,
                dy_o_norm,
                head_width=0.015,
                head_length=0.01,
                fc="green",
                ec="green",
                alpha=0.7,
                linewidth=2,
                zorder=4,
            )

        # Plot output trajectory
        ax.plot(
            player_output["x_norm"],
            player_output["y_norm"],
            "s--",
            color=color,
            alpha=0.9,
            linewidth=2,
            markersize=6,
            label="Output",
        )

        # Mark origin
        ax.plot(
            0,
            0,
            "r*",
            markersize=15,
            label="Origin",
            zorder=10,
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # Add reference lines
        ax.axhline(y=0, color="k", linestyle=":", alpha=0.3, linewidth=1)
        ax.axvline(x=0, color="k", linestyle=":", alpha=0.3, linewidth=1)

        ax.set_xlabel("X_norm", fontsize=10)
        ax.set_ylabel("Y_norm", fontsize=10)
        title = f"Player {nfl_id}"
        if is_target:
            title += " [TARGET]"
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=7, framealpha=0.9)

    # Main title
    fig.suptitle(
        f"Trajectory Comparison - Game {game_id}, Play {play_id}\n"
        + f"Reconstruction Error: {reconstruction_error:.6f} yards",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(
        f"  ✓ Saved: {output_path.name} ({n_players} players, error: {reconstruction_error:.6f} yards)"
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize normalized trajectories")
    parser.add_argument(
        "--fold",
        type=str,
        default="fold1",
        choices=["fold1", "fold2"],
        help="Which fold to visualize",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Which split to visualize",
    )
    parser.add_argument(
        "--n_samples", type=int, default=6, help="Number of sample plays to visualize"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/data_viz",
        help="Directory to save visualizations",
    )

    args = parser.parse_args()

    # Setup paths
    fold_dir = Path("data/processed") / args.fold
    output_dir = Path(args.output_dir) / args.fold
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Visualizing Normalized Trajectories")
    print(f"{'='*60}")
    print(f"Fold: {args.fold}")
    print(f"Split: {args.split}")
    print(f"Samples: {args.n_samples}\n")

    # Load data
    print("Loading data...")
    input_df, output_df, sample_plays = load_sample_plays(
        fold_dir, args.split, args.n_samples
    )
    print(f"Loaded {len(input_df):,} input rows, {len(output_df):,} output rows")
    print(f"Visualizing {len(sample_plays)} sample plays\n")

    # Process each play
    for idx, (_, row) in enumerate(sample_plays.iterrows(), 1):
        game_id = row["game_id"]
        play_id = row["play_id"]

        print(
            f"[{idx}/{len(sample_plays)}] Processing Game {game_id}, Play {play_id}..."
        )

        # Normalize play
        play_input, play_output, reconstruction_error = normalize_play(
            input_df, output_df, game_id, play_id
        )

        if play_input is None:
            print(f"  ⚠ Skipping - no data found")
            continue

        # Plot comparison
        output_path = (
            output_dir / f"{args.split}_comparison_game{game_id}_play{play_id}.png"
        )
        plot_comparison_trajectory(
            play_input, play_output, game_id, play_id, reconstruction_error, output_path
        )

    print(f"\n{'='*60}")
    print(f"✓ Visualization complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
