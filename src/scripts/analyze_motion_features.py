"""Analyze direction, orientation, speed, and acceleration columns in tracking data.

The script:
    * Summarizes definitions for `dir` and `o`
    * Computes descriptive statistics and percentiles for `dir`, `o`, `s`, and `a`
    * Generates histograms for each feature
    * Writes consolidated findings to `reports/motion_analysis/`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


FEATURE_DEFINITIONS = {
    "dir": (
        "Angle of player motion in degrees, measured clockwise from the positive y-axis "
        "(0° points toward the opponent's end zone when facing 'north')."
    ),
    "o": (
        "Orientation of the player's body in degrees, measured clockwise from the "
        "positive y-axis (0° means facing directly 'upfield')."
    ),
    "s": "Player speed in yards/second based on tracking displacements.",
    "a": "Player acceleration in yards/second² (time derivative of speed).",
}


def discover_files(data_dir: Path, pattern: str) -> list[Path]:
    files = sorted(data_dir.glob(pattern))
    if not files:
        msg = f"No files matching '{pattern}' found in {data_dir}"
        raise FileNotFoundError(msg)
    return files


def load_features(files: Iterable[Path], columns: list[str]) -> pl.DataFrame:
    lazy_frames = [pl.scan_csv(path).select(columns) for path in files]
    return pl.concat(lazy_frames).collect(streaming=True)


def compute_percentiles(series: np.ndarray, percentiles: list[float]) -> dict[str, float]:
    values = np.percentile(series, percentiles)
    return {f"p{int(p)}": float(v) for p, v in zip(percentiles, values, strict=False)}


def summarize_feature(series: pl.Series, circular: bool = False) -> dict[str, float]:
    data = series.drop_nulls().to_numpy()
    summary = {
        "count": int(data.size),
        "min": float(data.min()) if data.size else float("nan"),
        "max": float(data.max()) if data.size else float("nan"),
        "mean": float(data.mean()) if data.size else float("nan"),
        "std": float(data.std(ddof=0)) if data.size else float("nan"),
    }

    if data.size:
        summary.update(compute_percentiles(data, percentiles=[25, 50, 75, 90, 95, 99]))

    if circular and data.size:
        radians = np.deg2rad(data % 360)
        resultant = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
        circular_mean = np.rad2deg(resultant) % 360
        summary["circular_mean"] = float(circular_mean)

    return summary


def plot_histogram(
    data: np.ndarray,
    output_path: Path,
    *,
    bins: int = 60,
    range_: tuple[float, float] | None = None,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=bins, range=range_, edgecolor="black", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_markdown_report(
    output_path: Path,
    definitions: dict[str, str],
    stats: dict[str, dict[str, float]],
) -> None:
    lines = [
        "# Motion Feature Analysis",
        "",
        "## Definitions",
    ]
    for feature, description in definitions.items():
        lines.append(f"- **{feature}**: {description}")
    lines.append("")
    lines.append("## Summary Statistics")
    for feature, metrics in stats.items():
        lines.append(f"### {feature}")
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.2f}")
            else:
                lines.append(f"- {key}: {value}")
        lines.append("")

    output_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze direction/orientation and speed/acceleration distributions."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/train"),
        help="Directory containing training CSV files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="input_*.csv",
        help="Glob pattern selecting pre-throw tracking files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/motion_analysis"),
        help="Directory where reports and plots are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_files(args.data_dir, args.pattern)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = ["dir", "o", "s", "a"]
    df = load_features(files, columns)

    stats: dict[str, dict[str, float]] = {}
    plot_specs = {
        "dir": {
            "title": "Direction Angle Distribution",
            "xlabel": "dir (degrees, clockwise from +y)",
            "bins": 72,
            "range": (0, 360),
            "circular": True,
        },
        "o": {
            "title": "Orientation Angle Distribution",
            "xlabel": "o (degrees, clockwise from +y)",
            "bins": 72,
            "range": (0, 360),
            "circular": True,
        },
        "s": {
            "title": "Speed Distribution",
            "xlabel": "Speed s (yards/second)",
            "bins": 60,
            "range": None,
            "circular": False,
        },
        "a": {
            "title": "Acceleration Distribution",
            "xlabel": "Acceleration a (yards/second²)",
            "bins": 80,
            "range": None,
            "circular": False,
        },
    }

    for feature, spec in plot_specs.items():
        series = df[feature]
        stats[feature] = summarize_feature(series, circular=spec["circular"])
        data = series.drop_nulls().to_numpy()
        if data.size:
            plot_path = output_dir / f"{feature}_hist.png"
            plot_histogram(
                data,
                plot_path,
                bins=spec["bins"],
                range_=spec["range"],
                title=spec["title"],
                xlabel=spec["xlabel"],
            )

    summary_path = output_dir / "motion_feature_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "definitions": FEATURE_DEFINITIONS,
                "statistics": stats,
                "source_files": [file.name for file in files],
            },
            indent=2,
        )
    )

    markdown_path = output_dir / "motion_feature_report.md"
    write_markdown_report(markdown_path, FEATURE_DEFINITIONS, stats)

    print("Analysis complete.")
    print(f"Summary JSON  : {summary_path}")
    print(f"Markdown report: {markdown_path}")
    print("Histograms saved:")
    for feature in plot_specs:
        path = output_dir / f"{feature}_hist.png"
        if path.exists():
            print(f"  - {path}")


if __name__ == "__main__":
    main()
