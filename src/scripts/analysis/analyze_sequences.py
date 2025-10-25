"""Analyze tracking sequence lengths to guide seq2seq encoder/decoder design.

The script scans training CSVs, computes distribution statistics for the number
of frames per (game_id, play_id, nfl_id), and produces plots and textual
recommendations for both encoder (pre-throw) and decoder (post-throw) sequence
length choices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def discover_input_files(data_dir: Path, pattern: str) -> list[Path]:
    files = sorted(data_dir.glob(pattern))
    if not files:
        available = sorted(str(p.name) for p in data_dir.glob("*.csv"))
        message = (
            f"No CSV files matched pattern '{pattern}' in {data_dir}. "
            f"Available files: {available}"
        )
        raise FileNotFoundError(message)
    return files


def aggregate_sequence_lengths(files: Iterable[Path]) -> pl.DataFrame:
    lazy_frames = []
    for csv_path in files:
        raw = pl.scan_csv(csv_path)
        available_cols = set(raw.columns)

        aggregation = [
            pl.count().alias("sequence_length"),
            pl.col("frame_id").max().alias("max_frame_id"),
        ]
        if "player_to_predict" in available_cols:
            aggregation.append(pl.col("player_to_predict").max().alias("player_to_predict"))
        if "num_frames_output" in available_cols:
            aggregation.append(pl.col("num_frames_output").max().alias("num_frames_output"))
        if "play_direction" in available_cols:
            aggregation.append(pl.first("play_direction").alias("play_direction"))

        lf = raw.group_by("game_id", "play_id", "nfl_id").agg(*aggregation)
        lf = lf.with_columns(pl.lit(csv_path.name).alias("source_file"))
        lazy_frames.append(lf)

    return pl.concat(lazy_frames).collect(streaming=True)


def compute_percentiles(lengths: np.ndarray, percentiles: Iterable[float]) -> Mapping[float, float]:
    percentile_values = np.percentile(lengths, list(percentiles), method="nearest")
    return dict(zip(percentiles, percentile_values, strict=False))


def save_histogram(lengths: np.ndarray, output_dir: Path, label: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = min(60, max(int(lengths.max()) - int(lengths.min()) + 1, 10))
    ax.hist(lengths, bins=bins, edgecolor="black", alpha=0.75)
    ax.set_xlabel("Pre-throw sequence length (frames)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {label} sequence lengths")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    output_path = output_dir / f"{label}_sequence_length_hist.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_cdf(lengths: np.ndarray, output_dir: Path, label: str) -> Path:
    sorted_lengths = np.sort(lengths)
    cumulative = np.linspace(0, 1, sorted_lengths.size, endpoint=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_lengths, cumulative, marker=".", linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Pre-throw sequence length (frames)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"Empirical CDF of {label} sequence lengths")
    ax.grid(True, linestyle="--", alpha=0.3)
    output_path = output_dir / f"{label}_sequence_length_cdf.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def format_percentiles(percentiles: Mapping[float, float]) -> str:
    lines = ["Percentile snapshot:"]
    for p in sorted(percentiles):
        value = int(np.ceil(percentiles[p]))
        lines.append(f"  p{p:>5.1f}: {value:>4d} frames")
    return "\n".join(lines)


def recommend_hyperparameters(
    percentiles: Mapping[float, float], recommendation_kind: str
) -> dict[str, int]:
    ninety = int(np.ceil(percentiles.get(90.0, 0)))
    ninety_five = int(np.ceil(percentiles.get(95.0, ninety)))
    max_candidate = int(np.ceil(percentiles.get(99.0, ninety_five)))
    prefix = "encoder" if recommendation_kind == "encoder" else "decoder"
    return {
        f"{prefix}_sequence_length": ninety,
        f"{prefix}_padding_length": ninety_five,
        f"{prefix}_hard_cap": max_candidate,
    }


def print_summary(
    label: str,
    df: pl.DataFrame,
    percentiles: Mapping[float, float],
    recommendations: Mapping[str, int],
) -> None:
    lengths = df["sequence_length"].to_numpy()
    stats = {
        "total_sequences": int(df.height),
        "min_frames": int(lengths.min()),
        "max_frames": int(lengths.max()),
        "mean_frames": float(lengths.mean()),
        "median_frames": float(np.median(lengths)),
    }

    print(f"=== {label.capitalize()} Sequence Length Analysis ===")
    print(f"Sequences analyzed : {stats['total_sequences']}")
    print(f"Frame count range  : {stats['min_frames']} - {stats['max_frames']}")
    print(f"Mean / median      : {stats['mean_frames']:.2f} / {stats['median_frames']:.2f}")
    print(format_percentiles(percentiles))
    print()
    print("Recommended hyperparameters:")
    for key, value in recommendations.items():
        print(f"  {key}: {value}")
    print()
    print("Rationale:")
    if "encoder_sequence_length" in recommendations:
        print(
            "  • encoder_sequence_length at the 90th percentile keeps padding low while covering most plays."
        )
        print(
            "  • encoder_padding_length at ~95th percentile reduces truncation risk in evaluation."
        )
        print(
            "  • encoder_hard_cap bounds bucketing or clipping exceptionally long plays."
        )
    else:
        print(
            "  • decoder_sequence_length near the 90th percentile balances coverage and computation."
        )
        print(
            "  • decoder_padding_length around 95th percentile minimizes truncation."
        )
        print(
            "  • decoder_hard_cap ensures rare long trajectories are handled consistently."
        )


def persist_metadata(
    output_dir: Path,
    label: str,
    percentiles: Mapping[float, float],
    histogram_path: Path,
    cdf_path: Path,
    recommendations: Mapping[str, int],
) -> Path:
    metadata = {
        "label": label,
        "percentiles": {f"p{int(k)}": float(v) for k, v in percentiles.items()},
        "plots": {
            "histogram": str(histogram_path),
            "cdf": str(cdf_path),
        },
        "recommendations": recommendations,
    }
    metadata_path = output_dir / f"{label}_sequence_length_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata_path


def run_analysis(
    label: str,
    files: Iterable[Path],
    output_dir: Path,
    percentiles: Iterable[float],
    recommendation_kind: str,
) -> dict[str, object]:
    df = aggregate_sequence_lengths(files)
    lengths = df["sequence_length"].to_numpy()
    percentiles_map = compute_percentiles(lengths, percentiles)
    histogram_path = save_histogram(lengths, output_dir, label)
    cdf_path = save_cdf(lengths, output_dir, label)
    recommendations = recommend_hyperparameters(percentiles_map, recommendation_kind)
    summary_path = persist_metadata(
        output_dir,
        label,
        percentiles_map,
        histogram_path,
        cdf_path,
        recommendations,
    )
    print_summary(label, df, percentiles_map, recommendations)
    return {
        "label": label,
        "summary_path": summary_path,
        "histogram": histogram_path,
        "cdf": cdf_path,
        "recommendations": recommendations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pre-throw tracking sequence length distributions."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/train"),
        help="Directory containing pre-throw CSV files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="input_*.csv",
        help="Glob pattern for pre-throw CSV files within data-dir.",
    )
    parser.add_argument(
        "--decoder-pattern",
        type=str,
        default="output_*.csv",
        help="Glob pattern for post-throw CSV files within data-dir.",
    )
    parser.add_argument(
        "--skip-decoder",
        action="store_true",
        help="Skip decoder (post-throw) analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/sequence_analysis"),
        help="Directory to store generated plots and metadata.",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[50.0, 75.0, 90.0, 95.0, 97.5, 99.0],
        help="Percentiles to compute for the sequence length distribution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running encoder (pre-throw) analysis...")
    encoder_files = discover_input_files(args.data_dir, args.pattern)
    encoder_result = run_analysis(
        label="encoder",
        files=encoder_files,
        output_dir=output_dir,
        percentiles=args.percentiles,
        recommendation_kind="encoder",
    )
    print(f"Histogram saved to : {encoder_result['histogram']}")
    print(f"CDF saved to       : {encoder_result['cdf']}")
    print(f"Summary metadata   : {encoder_result['summary_path']}")

    aggregate_results = {"encoder": encoder_result["recommendations"]}

    if not args.skip_decoder:
        print("\nRunning decoder (post-throw) analysis...")
        decoder_files = discover_input_files(args.data_dir, args.decoder_pattern)
        decoder_result = run_analysis(
            label="decoder",
            files=decoder_files,
            output_dir=output_dir,
            percentiles=args.percentiles,
            recommendation_kind="decoder",
        )
        print(f"Histogram saved to : {decoder_result['histogram']}")
        print(f"CDF saved to       : {decoder_result['cdf']}")
        print(f"Summary metadata   : {decoder_result['summary_path']}")
        aggregate_results["decoder"] = decoder_result["recommendations"]

    combined_path = output_dir / "sequence_length_recommendations.json"
    combined_path.write_text(json.dumps(aggregate_results, indent=2))
    print(f"\nCombined recommendations written to: {combined_path}")


if __name__ == "__main__":
    main()
