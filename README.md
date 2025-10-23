# NFL Data Bowl 2025 Baselines

## Overview
- Frame-level tracking data provides pre-throw player trajectories plus ball landing coordinates; the task is to forecast select players after the throw.
- We prioritize compact, reproducible baselines before exploring larger attention-based models.
- Dashboards and notebooks remain the entry point for qualitative review of trajectories and error patterns.

## Approach 1 – Seq2Seq Trajectory Baseline
- Grounded in `src/appraoch1.md`, this baseline treats every `(game_id, play_id, nfl_id)` triple as an independent sequence-to-sequence example.
- Pre-throw frames are padded to a fixed encoder length (≈90th percentile) with masks passed to the model.
- Coordinates are re-centered on the final pre-throw position, rotated so offenses move left-to-right, and scaled by field dimensions; orientation and direction angles follow the same rotation to keep features consistent.
- Folded train/validation splits use late-season (weeks 15–18) and mid-season (weeks 9–12) holds to monitor temporal generalization.

### Model Implementation (`src/model.py`)
- `_mask_to_lengths` converts boolean frame masks into packed-sequence lengths so encoders ignore padding while guaranteeing ≥1 step.
- `BaseRecurrentEncoder` wraps a Torch RNN, handling packing/unpacking; `LSTMEncoder` and `GRUEncoder` configure LSTM/GRU stacks with dropout stripped when there is only one layer.
- `BaseDecoder` iteratively unrolls the decoder, exposing teacher forcing, a learnable feedback projection, and an output head; `LSTMDecoder`/`GRUDecoder` are thin wrappers around the respective cells.
- `Seq2SeqTrajectoryModel` connects the encoder/decoder pair: it encodes masked pre-throw frames, decodes future positions with the requested teacher forcing ratio, and exposes `predict` for auto-regressive rollout during inference.

### Next Steps
- Hook the baseline into Lightning training loops once dataset loaders are finalized.
- Compare LSTM and GRU backbones, then extend with convolutional encoders once normalization is validated.
