# Approach 3 – Target Receiver Pursuit TODO

## Immediate
- [ ] Run baseline training with `config_approach3_target.py` on fold1; record target-only RMSE vs Approach 1.
- [ ] Validate inference fallback: ensure non-target players hold last positions in dashboard overlay.
- [ ] Generate qualitative plots (distance-to-ball, trajectory overlays) comparing predictions vs ground truth for top/bottom plays.

## Experiments
- [ ] Try longer decoder horizon (40 frames) to cover extended air times; monitor stability.
- [ ] Add ball-relative features (distance, bearing, normalized closing speed) directly to dataset encoder inputs.
- [ ] Evaluate alternative fallback (linear glide to ball landing) for non-target players and measure dashboard interpretability.

## Integration
- [ ] Update dashboard pipeline to label target vs non-target predictions, highlighting fallback usage.
- [ ] Create wandb template for target-only runs with key metrics (target RMSE, pursuit ratio, final distance).
- [ ] Draft write-up summarizing Approach 3 motivation, architecture, and early results for the team repo.
