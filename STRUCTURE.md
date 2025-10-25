# Repository Structure

## Directory Overview

```
nfl-data-bowl-2025/
├── src/                          # Source code
│   ├── data.py                   # PyTorch Dataset classes
│   ├── model.py                  # Model architectures
│   ├── trainer.py                # Training loop
│   ├── inference.py              # Inference utilities
│   ├── configs/                  # Training configurations
│   ├── utils/                    # Utility functions (metrics, normalization, parsers)
│   └── scripts/
│       ├── train.py              # Training entry point
│       ├── pretrain_next_frame.py
│       ├── prepare_data.py
│       ├── test_dataset.py
│       └── analysis/             # Analysis and visualization scripts
│           ├── analyze_kalman_filter.py
│           ├── analyze_motion_features.py
│           ├── analyze_sequences.py
│           ├── analyze_speed_acceleration.py
│           ├── visualize_dataset_variants.py
│           └── visualize_normalized_data.py
│
├── apps/                         # Application scripts
│   ├── dashboard.py              # Streamlit dashboard
│   ├── evaluate_checkpoint.py    # Model evaluation
│   ├── evaluate_validation.py
│   └── generate_dashboard_predictions.py
│
├── notebooks/                    # Jupyter notebooks
│   ├── 0_eda.ipynb              # Exploratory data analysis
│   └── base_model.ipynb         # Base model experiments
│
├── docs/                         # Documentation
│   ├── AGENTS.md                # Agent documentation
│   ├── Data_description.md      # Data description
│   └── planning/                # Planning documents
│       ├── approach1_todo.md
│       ├── approach2.md
│       ├── approach3_todo.md
│       └── todo.md
│
├── tests/                        # Unit tests
│   ├── test_data.py
│   ├── test_metrics.py
│   ├── test_model.py
│   └── test_normalization.py
│
├── data/                         # Data files
│   ├── train/                   # Training data
│   ├── processed/               # Processed data
│   └── *.csv                    # Test data
│
├── outputs/                      # Generated outputs
│   ├── predictions/             # Prediction CSVs
│   ├── submissions/             # Competition submissions
│   └── validation/              # Validation predictions
│
├── checkpoints/                  # Model checkpoints
│   ├── approach1_v0/
│   ├── approach1_v1/
│   └── approach3_target/
│
├── logs/                         # Training logs
│   └── approach*/
│
├── reports/                      # Analysis reports
│   ├── dataset_memory_usage.md
│   ├── data_viz/
│   ├── motion_analysis/
│   └── sequence_analysis/
│
└── plots/                        # Generated plots
```

## Quick Start

### Training
```bash
# Train a model
uv run python -m src.scripts.train --config src/configs/config_approach1_v0.py --fold fold1

# Pretrain next frame predictor
uv run python -m src.scripts.pretrain_next_frame
```

### Analysis
```bash
# Analyze Kalman filter hypothesis
uv run python src/scripts/analysis/analyze_kalman_filter.py

# Analyze motion features
uv run python src/scripts/analysis/analyze_motion_features.py
```

### Evaluation
```bash
# Evaluate checkpoint
uv run python apps/evaluate_checkpoint.py --checkpoint checkpoints/approach1_v1/best.ckpt

# Generate predictions
uv run python apps/generate_dashboard_predictions.py
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_data.py
```

## Key Files

- `src/data.py` - PyTorch Dataset for trajectory prediction
- `src/model.py` - Transformer-based trajectory prediction model
- `src/trainer.py` - PyTorch Lightning training module
- `src/inference.py` - Inference and prediction utilities
- `pyproject.toml` - Project dependencies and configuration
- `RESTRUCTURING_PLAN.md` - Detailed restructuring plan for future improvements

## Recent Changes

✅ **Completed Restructuring:**
1. Analysis scripts organized in `src/scripts/analysis/`
2. Prediction/submission files moved to `outputs/` directory
3. Documentation consolidated in `docs/` directory
4. Notebooks moved to `notebooks/` directory
5. Application scripts moved to `apps/` directory

## Next Steps

See `RESTRUCTURING_PLAN.md` for medium and low priority improvements:
- Further modularize `src/` with subdirectories (data, models, training, inference)
- Split model architectures by approach
- Update import paths after restructuring
