"""
Analyze Kalman filter hypothesis for NFL speed and acceleration computation.

Tests whether NFL applies Kalman filtering to:
1. Position tracking to get smoothed velocity
2. Velocity to get smoothed acceleration

Kalman filters are commonly used in sports tracking systems because they:
- Handle sensor noise optimally
- Provide real-time estimation
- Can incorporate motion models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import wilcoxon

# Set random seed for reproducibility
np.random.seed(42)


class KalmanFilter1D:
    """Simple 1D Kalman filter for position -> velocity -> acceleration."""

    def __init__(
        self,
        process_variance,
        measurement_variance,
        initial_value=0,
        initial_variance=1,
    ):
        self.process_variance = process_variance  # Q: how much we trust the model
        self.measurement_variance = (
            measurement_variance  # R: how much we trust measurements
        )
        self.estimate = initial_value
        self.estimate_variance = initial_variance

    def update(self, measurement):
        """Update filter with new measurement."""
        # Prediction step (assume constant model)
        predicted_estimate = self.estimate
        predicted_variance = self.estimate_variance + self.process_variance

        # Update step
        kalman_gain = predicted_variance / (
            predicted_variance + self.measurement_variance
        )
        self.estimate = predicted_estimate + kalman_gain * (
            measurement - predicted_estimate
        )
        self.estimate_variance = (1 - kalman_gain) * predicted_variance

        return self.estimate


def kalman_filter_position_to_velocity(x, y, process_var, measurement_var, dt=0.1):
    """
    Apply Kalman filter to position to get velocity.
    Returns vx, vy, speed.
    """
    # Compute raw velocities
    raw_vx = np.gradient(x, dt)
    raw_vy = np.gradient(y, dt)

    # Apply Kalman filter to velocities
    kf_vx = KalmanFilter1D(process_var, measurement_var, initial_value=raw_vx[0])
    kf_vy = KalmanFilter1D(process_var, measurement_var, initial_value=raw_vy[0])

    filtered_vx = np.array([kf_vx.update(v) for v in raw_vx])
    filtered_vy = np.array([kf_vy.update(v) for v in raw_vy])

    speed = np.sqrt(filtered_vx**2 + filtered_vy**2)

    return filtered_vx, filtered_vy, speed


def kalman_filter_velocity_to_acceleration(
    vx, vy, process_var, measurement_var, dt=0.1
):
    """
    Apply Kalman filter to velocity to get acceleration.
    Returns ax, ay, acceleration magnitude.
    """
    # Compute raw accelerations
    raw_ax = np.gradient(vx, dt)
    raw_ay = np.gradient(vy, dt)

    # Apply Kalman filter to accelerations
    kf_ax = KalmanFilter1D(process_var, measurement_var, initial_value=raw_ax[0])
    kf_ay = KalmanFilter1D(process_var, measurement_var, initial_value=raw_ay[0])

    filtered_ax = np.array([kf_ax.update(a) for a in raw_ax])
    filtered_ay = np.array([kf_ay.update(a) for a in raw_ay])

    accel = np.sqrt(filtered_ax**2 + filtered_ay**2)

    return filtered_ax, filtered_ay, accel


def optimize_kalman_params(x, y, true_speed, true_accel, skip_frames=2):
    """
    Optimize Kalman filter parameters for both velocity and acceleration.
    Returns (q_vel, r_vel, q_accel, r_accel).
    """

    def objective(params):
        q_vel, r_vel, q_accel, r_accel = params

        # Ensure positive values
        q_vel = abs(q_vel)
        r_vel = abs(r_vel)
        q_accel = abs(q_accel)
        r_accel = abs(r_accel)

        # Apply Kalman filter to get velocity
        vx, vy, pred_speed = kalman_filter_position_to_velocity(x, y, q_vel, r_vel)

        # Apply Kalman filter to get acceleration
        ax, ay, pred_accel = kalman_filter_velocity_to_acceleration(
            vx, vy, q_accel, r_accel
        )

        # Compute combined error
        valid = slice(skip_frames, -skip_frames if skip_frames > 0 else None)
        speed_rmse = np.sqrt(np.mean((pred_speed[valid] - true_speed[valid]) ** 2))
        accel_rmse = np.sqrt(np.mean((pred_accel[valid] - true_accel[valid]) ** 2))

        # Weight acceleration more since that's what we care about
        return accel_rmse + 0.1 * speed_rmse

    # Grid search for initial guess
    best_score = float("inf")
    best_params = (0.1, 0.1, 0.1, 0.1)

    for q_vel in [0.01, 0.1, 1.0]:
        for r_vel in [0.01, 0.1, 1.0]:
            for q_accel in [0.01, 0.1, 1.0]:
                for r_accel in [0.01, 0.1, 1.0]:
                    score = objective((q_vel, r_vel, q_accel, r_accel))
                    if score < best_score:
                        best_score = score
                        best_params = (q_vel, r_vel, q_accel, r_accel)

    return best_params


def main():
    print("Loading data...")
    data_path = Path("data/train")

    # Load tracking data (input files contain position/speed/acceleration)
    tracking_files = sorted(data_path.glob("input_2023_*.csv"))
    if not tracking_files:
        print("No tracking files found!")
        return

    # Load first tracking file
    df = pd.read_csv(tracking_files[0])

    # Get unique game-play-nfl combinations
    trajectories = df.groupby(["game_id", "play_id", "nfl_id"])
    print(f"Found {len(trajectories)} trajectories")

    # Sample trajectories for analysis
    n_samples = 100
    sample_indices = np.random.choice(
        len(trajectories), size=min(n_samples, len(trajectories)), replace=False
    )

    results = []

    print(f"\nAnalyzing Kalman filter hypothesis for {n_samples} trajectories...")
    for idx, (name, group) in enumerate(trajectories):
        if idx not in sample_indices:
            continue

        if idx % 10 == 0:
            print(f"Processing {name}/{n_samples}...")

        # Sort by frame
        group = group.sort_values("frame_id").reset_index(drop=True)

        # Need at least 10 frames
        if len(group) < 10:
            continue

        x = group["x"].values
        y = group["y"].values
        true_speed = group["s"].values
        true_accel = group["a"].values

        # Optimize Kalman parameters
        q_vel_opt, r_vel_opt, q_accel_opt, r_accel_opt = optimize_kalman_params(
            x, y, true_speed, true_accel, skip_frames=2
        )

        # Compute with optimal parameters
        vx_opt, vy_opt, pred_speed_opt = kalman_filter_position_to_velocity(
            x, y, q_vel_opt, r_vel_opt
        )
        ax_opt, ay_opt, pred_accel_opt = kalman_filter_velocity_to_acceleration(
            vx_opt, vy_opt, q_accel_opt, r_accel_opt
        )

        # Compute with fixed parameters (typical values)
        vx_fixed, vy_fixed, pred_speed_fixed = kalman_filter_position_to_velocity(
            x, y, 0.1, 0.1
        )
        ax_fixed, ay_fixed, pred_accel_fixed = kalman_filter_velocity_to_acceleration(
            vx_fixed, vy_fixed, 0.1, 0.1
        )

        # Skip first and last 2 frames for metrics
        valid = slice(2, -2)

        # Metrics for optimal
        speed_rmse_opt = np.sqrt(
            np.mean((pred_speed_opt[valid] - true_speed[valid]) ** 2)
        )
        speed_corr_opt = np.corrcoef(pred_speed_opt[valid], true_speed[valid])[0, 1]
        accel_rmse_opt = np.sqrt(
            np.mean((pred_accel_opt[valid] - true_accel[valid]) ** 2)
        )
        accel_corr_opt = np.corrcoef(pred_accel_opt[valid], true_accel[valid])[0, 1]

        # Metrics for fixed
        speed_rmse_fixed = np.sqrt(
            np.mean((pred_speed_fixed[valid] - true_speed[valid]) ** 2)
        )
        speed_corr_fixed = np.corrcoef(pred_speed_fixed[valid], true_speed[valid])[0, 1]
        accel_rmse_fixed = np.sqrt(
            np.mean((pred_accel_fixed[valid] - true_accel[valid]) ** 2)
        )
        accel_corr_fixed = np.corrcoef(pred_accel_fixed[valid], true_accel[valid])[0, 1]

        results.append(
            {
                "game_id": name[0],
                "play_id": name[1],
                "nfl_id": name[2],
                "n_frames": len(group),
                "q_vel_opt": q_vel_opt,
                "r_vel_opt": r_vel_opt,
                "q_accel_opt": q_accel_opt,
                "r_accel_opt": r_accel_opt,
                "speed_rmse_opt": speed_rmse_opt,
                "speed_corr_opt": speed_corr_opt,
                "accel_rmse_opt": accel_rmse_opt,
                "accel_corr_opt": accel_corr_opt,
                "speed_rmse_fixed": speed_rmse_fixed,
                "speed_corr_fixed": speed_corr_fixed,
                "accel_rmse_fixed": accel_rmse_fixed,
                "accel_corr_fixed": accel_corr_fixed,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("KALMAN FILTER HYPOTHESIS TEST")
    print("=" * 80)

    print("\nOptimal Kalman Parameters:")
    print(
        f"  Velocity Process Var (Q) - Mean: {results_df['q_vel_opt'].mean():.4f}, Median: {results_df['q_vel_opt'].median():.4f}"
    )
    print(
        f"  Velocity Measurement Var (R) - Mean: {results_df['r_vel_opt'].mean():.4f}, Median: {results_df['r_vel_opt'].median():.4f}"
    )
    print(
        f"  Accel Process Var (Q) - Mean: {results_df['q_accel_opt'].mean():.4f}, Median: {results_df['q_accel_opt'].median():.4f}"
    )
    print(
        f"  Accel Measurement Var (R) - Mean: {results_df['r_accel_opt'].mean():.4f}, Median: {results_df['r_accel_opt'].median():.4f}"
    )

    print("\nSpeed Prediction (with optimal Kalman params):")
    print(f"  RMSE - Mean:   {results_df['speed_rmse_opt'].mean():.4f} yards/sec")
    print(f"  RMSE - Median: {results_df['speed_rmse_opt'].median():.4f} yards/sec")
    print(f"  Corr - Mean:   {results_df['speed_corr_opt'].mean():.4f}")
    print(f"  Corr - Median: {results_df['speed_corr_opt'].median():.4f}")

    print("\nAcceleration Prediction (with optimal Kalman params):")
    print(f"  RMSE - Mean:   {results_df['accel_rmse_opt'].mean():.4f} yards/sec²")
    print(f"  RMSE - Median: {results_df['accel_rmse_opt'].median():.4f} yards/sec²")
    print(f"  Corr - Mean:   {results_df['accel_corr_opt'].mean():.4f}")
    print(f"  Corr - Median: {results_df['accel_corr_opt'].median():.4f}")

    print("\nAcceleration Prediction (with fixed Q=R=0.1):")
    print(f"  RMSE - Mean:   {results_df['accel_rmse_fixed'].mean():.4f} yards/sec²")
    print(f"  RMSE - Median: {results_df['accel_rmse_fixed'].median():.4f} yards/sec²")
    print(f"  Corr - Mean:   {results_df['accel_corr_fixed'].mean():.4f}")
    print(f"  Corr - Median: {results_df['accel_corr_fixed'].median():.4f}")

    # Compare with previous best methods
    print("\nComparison with Previous Best Methods:")
    print(f"  Position 2nd Deriv + EWMA: RMSE=0.9399, Corr=0.7588")
    print(f"  Double EWMA:               RMSE=0.9603, Corr=0.6477")
    print(
        f"  Kalman Filter (optimal):   RMSE={results_df['accel_rmse_opt'].mean():.4f}, Corr={results_df['accel_corr_opt'].mean():.4f}"
    )

    improvement_vs_ewma = (0.9399 - results_df["accel_rmse_opt"].mean()) / 0.9399 * 100
    print(f"  RMSE Change vs EWMA: {improvement_vs_ewma:+.1f}%")

    print("\n" + "=" * 80)

    # Save results
    output_dir = Path("reports/motion_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "kalman_filter_analysis_results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'kalman_filter_analysis_results.csv'}")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Velocity parameter distributions
    axes[0, 0].hist(
        results_df["q_vel_opt"], bins=20, alpha=0.7, label="Process Var (Q)"
    )
    axes[0, 0].hist(
        results_df["r_vel_opt"], bins=20, alpha=0.7, label="Measurement Var (R)"
    )
    axes[0, 0].axvline(
        results_df["q_vel_opt"].mean(), color="blue", linestyle="--", alpha=0.7
    )
    axes[0, 0].axvline(
        results_df["r_vel_opt"].mean(), color="orange", linestyle="--", alpha=0.7
    )
    axes[0, 0].set_xlabel("Variance")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Velocity Kalman Parameters")
    axes[0, 0].legend()
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(alpha=0.3)

    # Acceleration parameter distributions
    axes[0, 1].hist(
        results_df["q_accel_opt"], bins=20, alpha=0.7, label="Process Var (Q)"
    )
    axes[0, 1].hist(
        results_df["r_accel_opt"], bins=20, alpha=0.7, label="Measurement Var (R)"
    )
    axes[0, 1].axvline(
        results_df["q_accel_opt"].mean(), color="blue", linestyle="--", alpha=0.7
    )
    axes[0, 1].axvline(
        results_df["r_accel_opt"].mean(), color="orange", linestyle="--", alpha=0.7
    )
    axes[0, 1].set_xlabel("Variance")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Acceleration Kalman Parameters")
    axes[0, 1].legend()
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(alpha=0.3)

    # Q/R ratio
    q_r_ratio_vel = results_df["q_vel_opt"] / results_df["r_vel_opt"]
    q_r_ratio_accel = results_df["q_accel_opt"] / results_df["r_accel_opt"]
    axes[0, 2].hist(q_r_ratio_vel, bins=20, alpha=0.7, label="Velocity Q/R")
    axes[0, 2].hist(q_r_ratio_accel, bins=20, alpha=0.7, label="Accel Q/R")
    axes[0, 2].set_xlabel("Q/R Ratio")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title("Process/Measurement Variance Ratio")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # Speed RMSE comparison
    axes[1, 0].scatter(
        results_df["speed_rmse_fixed"], results_df["speed_rmse_opt"], alpha=0.5
    )
    max_val = max(
        results_df["speed_rmse_fixed"].max(), results_df["speed_rmse_opt"].max()
    )
    axes[1, 0].plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Equal")
    axes[1, 0].set_xlabel("Speed RMSE (Fixed)")
    axes[1, 0].set_ylabel("Speed RMSE (Optimal)")
    axes[1, 0].set_title("Speed RMSE: Fixed vs Optimal")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Accel RMSE comparison
    axes[1, 1].scatter(
        results_df["accel_rmse_fixed"], results_df["accel_rmse_opt"], alpha=0.5
    )
    max_val = max(
        results_df["accel_rmse_fixed"].max(), results_df["accel_rmse_opt"].max()
    )
    axes[1, 1].plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Equal")
    axes[1, 1].set_xlabel("Accel RMSE (Fixed)")
    axes[1, 1].set_ylabel("Accel RMSE (Optimal)")
    axes[1, 1].set_title("Accel RMSE: Fixed vs Optimal")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # Correlation distribution
    axes[1, 2].hist(results_df["accel_corr_opt"], bins=20, alpha=0.7, label="Optimal")
    axes[1, 2].hist(results_df["accel_corr_fixed"], bins=20, alpha=0.7, label="Fixed")
    axes[1, 2].axvline(
        0.7588, color="green", linestyle="--", alpha=0.7, label="EWMA (0.76)"
    )
    axes[1, 2].set_xlabel("Correlation")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].set_title("Acceleration Correlation Distribution")
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "kalman_filter_analysis.png", dpi=150, bbox_inches="tight")
    print(f"Visualization saved to {output_dir / 'kalman_filter_analysis.png'}")

    # Visualize a sample trajectory
    sample_idx = results_df["accel_rmse_opt"].idxmin()  # Best trajectory
    sample_row = results_df.iloc[sample_idx]

    # Get the trajectory
    sample_traj = trajectories.get_group(
        (sample_row["game_id"], sample_row["play_id"], sample_row["nfl_id"])
    )
    sample_traj = sample_traj.sort_values("frame_id").reset_index(drop=True)

    x = sample_traj["x"].values
    y = sample_traj["y"].values
    true_speed = sample_traj["s"].values
    true_accel = sample_traj["a"].values

    # Compute predictions
    vx, vy, pred_speed = kalman_filter_position_to_velocity(
        x, y, sample_row["q_vel_opt"], sample_row["r_vel_opt"]
    )
    ax, ay, pred_accel = kalman_filter_velocity_to_acceleration(
        vx, vy, sample_row["q_accel_opt"], sample_row["r_accel_opt"]
    )

    # Compute raw for comparison
    raw_vx = np.gradient(x, 0.1)
    raw_vy = np.gradient(y, 0.1)
    raw_speed = np.sqrt(raw_vx**2 + raw_vy**2)
    raw_ax = np.gradient(raw_vx, 0.1)
    raw_ay = np.gradient(raw_vy, 0.1)
    raw_accel = np.sqrt(raw_ax**2 + raw_ay**2)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    frames = np.arange(len(sample_traj))

    # Speed comparison
    axes[0].plot(frames, true_speed, "o-", label="NFL Provided", alpha=0.7, linewidth=2)
    axes[0].plot(frames, raw_speed, "s-", label="Raw (from positions)", alpha=0.5)
    axes[0].plot(
        frames,
        pred_speed,
        "^-",
        label=f'Kalman (Q={sample_row["q_vel_opt"]:.3f}, R={sample_row["r_vel_opt"]:.3f})',
        alpha=0.7,
    )
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Speed (yards/sec)")
    axes[0].set_title(
        f'Speed - RMSE={sample_row["speed_rmse_opt"]:.3f}, Corr={sample_row["speed_corr_opt"]:.3f}'
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Acceleration comparison
    axes[1].plot(frames, true_accel, "o-", label="NFL Provided", alpha=0.7, linewidth=2)
    axes[1].plot(frames, raw_accel, "s-", label="Raw (from Kalman vel)", alpha=0.5)
    axes[1].plot(
        frames,
        pred_accel,
        "^-",
        label=f'Kalman (Q={sample_row["q_accel_opt"]:.3f}, R={sample_row["r_accel_opt"]:.3f})',
        alpha=0.7,
    )
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Acceleration (yards/sec²)")
    axes[1].set_title(
        f'Acceleration - RMSE={sample_row["accel_rmse_opt"]:.3f}, Corr={sample_row["accel_corr_opt"]:.3f}'
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "kalman_filter_sample_trajectory.png", dpi=150, bbox_inches="tight"
    )
    print(
        f"Sample trajectory visualization saved to {output_dir / 'kalman_filter_sample_trajectory.png'}"
    )
    
    # Create scatter plots comparing actual vs computed for multiple samples
    print("\nCreating scatter plots for multiple samples...")
    
    # Select diverse samples (best, median, worst)
    sorted_by_rmse = results_df.sort_values('accel_rmse_opt')
    sample_indices = [
        sorted_by_rmse.index[0],  # Best
        sorted_by_rmse.index[len(sorted_by_rmse)//2],  # Median
        sorted_by_rmse.index[-1],  # Worst
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    for plot_idx, sample_idx in enumerate(sample_indices):
        sample_row = results_df.iloc[sample_idx]
        
        # Get the trajectory
        sample_traj = trajectories.get_group(
            (sample_row["game_id"], sample_row["play_id"], sample_row["nfl_id"])
        )
        sample_traj = sample_traj.sort_values("frame_id").reset_index(drop=True)
        
        x = sample_traj["x"].values
        y = sample_traj["y"].values
        true_speed = sample_traj["s"].values
        true_accel = sample_traj["a"].values
        
        # Compute predictions
        vx, vy, pred_speed = kalman_filter_position_to_velocity(
            x, y, sample_row["q_vel_opt"], sample_row["r_vel_opt"]
        )
        ax_vals, ay_vals, pred_accel = kalman_filter_velocity_to_acceleration(
            vx, vy, sample_row["q_accel_opt"], sample_row["r_accel_opt"]
        )
        
        # Skip first and last 2 frames
        valid = slice(2, -2)
        
        # Speed scatter plot
        axes[plot_idx, 0].scatter(true_speed[valid], pred_speed[valid], alpha=0.6, s=50)
        speed_min = min(true_speed[valid].min(), pred_speed[valid].min())
        speed_max = max(true_speed[valid].max(), pred_speed[valid].max())
        axes[plot_idx, 0].plot([speed_min, speed_max], [speed_min, speed_max], 'r--', alpha=0.5, linewidth=2, label='Perfect Match')
        axes[plot_idx, 0].set_xlabel('NFL Actual Speed (yards/sec)')
        axes[plot_idx, 0].set_ylabel('Kalman Computed Speed (yards/sec)')
        
        quality = ['Best', 'Median', 'Worst'][plot_idx]
        axes[plot_idx, 0].set_title(f'{quality} - Speed (RMSE={sample_row["speed_rmse_opt"]:.3f}, Corr={sample_row["speed_corr_opt"]:.3f})')
        axes[plot_idx, 0].legend()
        axes[plot_idx, 0].grid(alpha=0.3)
        
        # Acceleration scatter plot
        axes[plot_idx, 1].scatter(true_accel[valid], pred_accel[valid], alpha=0.6, s=50)
        accel_min = min(true_accel[valid].min(), pred_accel[valid].min())
        accel_max = max(true_accel[valid].max(), pred_accel[valid].max())
        axes[plot_idx, 1].plot([accel_min, accel_max], [accel_min, accel_max], 'r--', alpha=0.5, linewidth=2, label='Perfect Match')
        axes[plot_idx, 1].set_xlabel('NFL Actual Acceleration (yards/sec²)')
        axes[plot_idx, 1].set_ylabel('Kalman Computed Acceleration (yards/sec²)')
        axes[plot_idx, 1].set_title(f'{quality} - Acceleration (RMSE={sample_row["accel_rmse_opt"]:.3f}, Corr={sample_row["accel_corr_opt"]:.3f})')
        axes[plot_idx, 1].legend()
        axes[plot_idx, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(
        output_dir / "kalman_filter_actual_vs_computed.png", dpi=150, bbox_inches="tight"
    )
    print(f"Actual vs Computed scatter plots saved to {output_dir / 'kalman_filter_actual_vs_computed.png'}")


if __name__ == "__main__":
    main()
