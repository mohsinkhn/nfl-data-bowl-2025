# NFL Speed and Acceleration Computation Analysis

## Summary

NFL tracking data uses **Kalman filtering** to compute speed and acceleration from raw position measurements. Testing across 100 player trajectories shows Kalman filtering achieves **RMSE=0.64 yards/sec²** and **correlation=0.82** for acceleration prediction, significantly outperforming alternatives like EWMA smoothing (RMSE=0.94, 31.7% improvement). The optimal parameters suggest higher trust in velocity measurements (Q≈0.58, R≈0.22) and balanced filtering for acceleration (Q≈0.40, R≈0.49), consistent with real-time sports tracking systems using RFID/GPS sensors.

## Files

- `kalman_filter_analysis.png` - Parameter distributions and performance comparisons
- `kalman_filter_sample_trajectory.png` - Example trajectory showing Kalman predictions vs NFL provided values
- `kalman_filter_analysis_results.csv` - Detailed results for 100 trajectories
