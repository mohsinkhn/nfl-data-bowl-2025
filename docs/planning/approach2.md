# Approach 2 Plan – Polar (Δt, Δθ) Targets

## Motivation
- Predict relative motion (speed-like `Δt`, heading change `Δθ`) instead of absolute `(x, y)` to keep the learning space compact and rotation invariant.
- Inspired by Social-LSTM/Social-GAN (Alahi et al., CVPR 2016; Gupta et al., CVPR 2018), vehicle forecasting with kinematic embeddings (Li et al., ICRA 2020), and sports trajectory models using polar increments (Lu et al., KDD 2019).

## Data Pipeline
1. **Flag**: Add `use_polar_targets` in `DataConfig`; propagate through scripts.
2. **Target conversion**:
   - For each decoder step compute `Δx, Δy`, then `Δt = sqrt(Δx²+Δy²)` and `Δθ = wrap(atan2(Δy, Δx) − heading_prev)`.
   - Store cumulative headings and distances to allow reconstruction; metadata retains last encoder heading/position.
3. **Masks**: Keep existing `decoder_mask`; no change to padding logic.
4. **Inverse transform helper**: New utility to accumulate deltas and map back through `CoordinateTransform` for evaluation/output.

## Model Updates
- Decoder predicts `(Δt, Δθ)` (optionally sin/cos of `Δθ` to avoid discontinuity).
- Autoregressive integration: during forward/predict, accumulate deltas step-by-step to rebuild `(x, y)` before loss/metrics.
- Loss: MSE on `Δt`; angular loss via cosine distance or MSE on sin/cos.

## Evaluation/Inference
- Validation/inference scripts call the polar→Cartesian helper before RMSE logging or CSV generation.
- Retain physical-space RMSE (`val_rmse_physical`) for apples-to-apples comparison.

## Experiments
1. Baseline comparison: Cartesian vs polar targets with identical configs.
2. Ablate landing/role features in both parameterisations.
3. Optional extensions: Add acceleration term (`Δt` difference), or predict `(Δsin, Δcos)` directly.

## Implementation TODO
- [ ] `DataConfig`: add `use_polar_targets` flag.
- [ ] Dataset: when flag is true, return polar deltas + reconstruction metadata.
- [ ] Trainer/model: branch loss/prediction logic for polar mode.
- [ ] Utilities: helper to integrate polar outputs to Cartesian.
- [ ] Evaluation scripts: toggle based on checkpoint flag.

