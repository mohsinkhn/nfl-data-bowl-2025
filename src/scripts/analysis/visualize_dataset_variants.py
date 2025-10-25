"""Compare dataset preprocessing variants visually.

This script instantiates ``NFLTrajectoryDataset`` with different configuration
flags and plots the same play in raw vs canonical coordinates so we can see how
options such as normalization, rotation alignment, and residual targets affect
what the model receives.

Example usage:
    uv run python -m src.scripts.visualize_dataset_variants \
        --data-dir data/processed/fold1 \
        --split val \
        --sample-index 0 \
        --experiments default no_normalize no_rotation
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

# Allow running as a script without installing the package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import NFLTrajectoryDataset, RawSample  # type: ignore

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    name: str
    overrides: Dict[str, object]
    description: str


EXPERIMENTS: Dict[str, Experiment] = {
    "default": Experiment(
        name="default",
        overrides={},
        description="Normalize + rotation + heading alignment",
    ),
    "no_normalize": Experiment(
        name="no_normalize",
        overrides={"normalize": False},
        description="No normalization",
    ),
    "no_rotation": Experiment(
        name="no_rotation",
        overrides={"rotation_normalize": False},
        description="Disable play-direction rotation",
    ),
    "no_align_heading": Experiment(
        name="no_align_heading",
        overrides={"align_heading": False},
        description="Disable heading alignment",
    ),
    "residual_targets": Experiment(
        name="residual_targets",
        overrides={"use_ball_residual_targets": True},
        description="Ball-residual decoder targets",
    ),
}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _prepare_axes(fig: plt.Figure, title: str):
    ax_raw = fig.add_subplot(1, 2, 1)
    ax_can = fig.add_subplot(1, 2, 2)
    ax_raw.set_title(f"Raw coordinates\n{title}")
    ax_raw.set_xlabel("X (yards)")
    ax_raw.set_ylabel("Y (yards)")
    ax_raw.set_aspect("equal", adjustable="box")
    ax_raw.grid(True, linestyle="--", alpha=0.3)

    ax_can.set_title("Canonical coordinates")
    ax_can.set_xlabel("X (canonical)")
    ax_can.set_ylabel("Y (canonical)")
    ax_can.set_aspect("equal", adjustable="box")
    ax_can.grid(True, linestyle="--", alpha=0.3)
    return ax_raw, ax_can


def _plot_traj(ax, coords: np.ndarray, label: str, color: str, marker: str = "o"):
    if len(coords) == 0:
        return
    ax.plot(coords[:, 0], coords[:, 1], marker + "-", color=color, label=label)
    ax.scatter(coords[-1, 0], coords[-1, 1], color=color, edgecolor="black", s=60)


def visualise_experiment(
    dataset: NFLTrajectoryDataset,
    raw_sample: RawSample,
    output_dir: Path,
    experiment: Experiment,
    idx: int,
    show: bool,
) -> None:
    processed = dataset.processor.process(raw_sample)
    metadata = processed["metadata"]

    raw_input = raw_sample.input_features[:, :2]
    raw_output = raw_sample.output_positions
    ball_raw = raw_sample.ball_land

    transform = metadata.get("transform")
    canonical_output = metadata["decoder_target_cartesian"]
    if transform is not None:
        canonical_input = transform.transform_points(raw_input)
        canonical_output_plot = canonical_output
        ball_canonical = metadata["ball_land_canonical"]
    else:
        canonical_input = raw_input
        canonical_output_plot = raw_output
        ball_canonical = ball_raw

    figure_title = (
        f"{experiment.name} (sample #{idx})\n"
        f"normalize={dataset.processor.normalize}, "
        f"rotation={dataset.processor.rotation_normalize}, "
        f"align={dataset.processor.align_heading}, "
        f"residual={dataset.processor.use_ball_residual_targets}"
    )

    fig = plt.figure(figsize=(12, 5))
    ax_raw, ax_can = _prepare_axes(fig, figure_title)

    _plot_traj(ax_raw, raw_input, "Input", "tab:blue")
    _plot_traj(ax_raw, raw_output, "Output", "tab:red")
    ax_raw.scatter(ball_raw[0], ball_raw[1], marker="*", color="gold", s=120, label="Ball landing")
    ax_raw.legend()

    _plot_traj(ax_can, canonical_input, "Input", "tab:blue")
    _plot_traj(ax_can, canonical_output_plot, "Output", "tab:red")
    ax_can.scatter(ball_canonical[0], ball_canonical[1], marker="*", color="gold", s=120, label="Ball landing")
    ax_can.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{experiment.name}_sample{idx}.png"
    fig.tight_layout()
    fig.savefig(outfile)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved plot to {outfile}")


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dataset processing variants")
    parser.add_argument("--data-dir", type=str, default="data/processed/fold1", help="Path to processed fold directory")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val", help="Split to sample from")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index within dataset")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="*",
        default=["default", "no_normalize", "no_rotation", "no_align_heading", "residual_targets"],
        help="Experiment names to visualise",
    )
    parser.add_argument("--output-dir", type=str, default="plots/dataset_variants", help="Directory to save plots")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    input_parquet = data_dir / f"{args.split}_input.parquet"
    output_parquet = data_dir / f"{args.split}_output.parquet"

    base_kwargs = {
        "input_parquet": str(input_parquet),
        "output_parquet": str(output_parquet),
    }

    # Instantiate base dataset to pick a reference sample
    base_dataset = NFLTrajectoryDataset(**base_kwargs)
    if not base_dataset.samples:
        raise RuntimeError("Dataset is empty; check paths")

    sample_idx = args.sample_index % len(base_dataset.samples)
    reference_key = base_dataset.samples[sample_idx]
    print(f"Using sample #{sample_idx}: {reference_key}")

    output_dir = Path(args.output_dir)

    for name in args.experiments:
        if name not in EXPERIMENTS:
            print(f"⚠ Unknown experiment '{name}', skipping")
            continue

        exp = EXPERIMENTS[name]
        kwargs = base_kwargs.copy()
        kwargs.update(exp.overrides)
        dataset = NFLTrajectoryDataset(**kwargs)

        if reference_key not in dataset.input_groups:
            print(f"⚠ Experiment '{name}' missing sample {reference_key}, skipping")
            continue

        raw_sample = dataset._fetch_raw_sample(reference_key)
        print(f"Visualising experiment '{name}' -> {exp.description}")
        visualise_experiment(dataset, raw_sample, output_dir, exp, sample_idx, args.show)


if __name__ == "__main__":
    main()
