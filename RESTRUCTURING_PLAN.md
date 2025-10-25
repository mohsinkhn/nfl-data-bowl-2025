# Repository Structure Improvement Recommendations

## Current Issues

1. **Root directory clutter**: Multiple evaluation/prediction scripts at root level
2. **Mixed concerns**: Analysis, training, and inference scripts scattered
3. **Inconsistent naming**: Some files use underscores, some have typos (`appraoch1.md`)
4. **No clear separation**: Training vs inference vs analysis vs utilities

## Recommended Structure

```
nfl-data-bowl-2025/
├── src/
│   ├── data/                      # Data loading and processing
│   │   ├── __init__.py
│   │   ├── dataset.py            # Main PyTorch Dataset classes (rename from data.py)
│   │   ├── preprocessing.py      # Data cleaning, feature engineering
│   │   └── augmentation.py       # Data augmentation (if needed)
│   │
│   ├── models/                    # Model architectures
│   │   ├── __init__.py
│   │   ├── base.py               # Base model components (rename from model.py)
│   │   ├── approach1.py          # Approach-specific architectures
│   │   ├── approach2.py
│   │   └── approach3.py
│   │
│   ├── training/                  # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py            # Move trainer.py here
│   │   └── callbacks.py          # Custom callbacks (if any)
│   │
│   ├── inference/                 # Inference and prediction
│   │   ├── __init__.py
│   │   ├── predictor.py          # Move inference.py here, rename
│   │   ├── evaluate.py           # Move evaluate_checkpoint.py, evaluate_validation.py here
│   │   └── generate_submission.py # Submission generation logic
│   │
│   ├── utils/                     # Utilities (already good)
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── normalization.py
│   │   └── parsers.py
│   │
│   ├── configs/                   # Configuration files (already good)
│   │   ├── __init__.py
│   │   ├── config_approach1_v0.py
│   │   └── config_approach3_target.py
│   │
│   └── scripts/                   # Executable scripts
│       ├── __init__.py
│       ├── train.py              # Training entry point
│       ├── pretrain_next_frame.py
│       ├── prepare_data.py
│       ├── test_dataset.py
│       │
│       └── analysis/              # Analysis and visualization (DONE)
│           ├── __init__.py
│           ├── analyze_kalman_filter.py
│           ├── analyze_motion_features.py
│           ├── analyze_sequences.py
│           ├── analyze_speed_acceleration.py
│           ├── visualize_dataset_variants.py
│           └── visualize_normalized_data.py
│
├── notebooks/                     # Move notebooks here
│   ├── 0_eda.ipynb
│   └── base_model.ipynb
│
├── apps/                          # Application scripts (dashboard, etc.)
│   ├── dashboard.py
│   └── generate_dashboard_predictions.py
│
├── docs/                          # Documentation
│   ├── AGENTS.md                 # Move from root
│   ├── Data_description.md       # Move from root
│   ├── approach1.md              # Fix typo, move from src/
│   ├── approach2.md              # Move from root
│   └── planning/                 # Planning documents
│       ├── approach1_todo.md
│       ├── approach3_todo.md
│       └── todo.md
│
├── tests/                         # Tests (already good structure)
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_metrics.py
│   ├── test_model.py
│   └── test_normalization.py
│
├── data/                          # Data directory (already good)
│   ├── train/
│   ├── processed/
│   └── *.csv
│
├── outputs/                       # NEW: Consolidate outputs
│   ├── predictions/              # All prediction CSVs
│   │   ├── predictions_week15.csv
│   │   ├── predictions_week15_for_dashboard.csv
│   │   └── ...
│   ├── submissions/              # Competition submissions
│   │   ├── submission_approach1_v1.csv
│   │   └── submission_approach1_v2.csv
│   └── validation/               # Validation predictions
│       └── val_predictions_*.csv
│
├── checkpoints/                   # Model checkpoints (already good)
│   ├── approach1_v0/
│   ├── approach1_v1/
│   └── approach3_target/
│
├── logs/                          # Training logs (already good)
│   ├── approach1_v0/
│   └── approach1_v1/
│
├── reports/                       # Analysis reports (already good)
│   ├── dataset_memory_usage.md
│   ├── data_viz/
│   ├── motion_analysis/
│   └── sequence_analysis/
│
├── plots/                         # Generated plots (keep)
│
├── wandb/                         # Wandb runs (keep)
│
├── .coverage                      # Coverage report (keep)
├── .pytest_cache/                 # Pytest cache (keep)
├── .venv/                         # Virtual env (keep)
├── pyproject.toml                 # Project config (keep)
├── uv.lock                        # Lock file (keep)
├── README.md                      # Main readme (keep)
└── package.json                   # Node packages (keep)
```

## Priority Actions

### High Priority (Do Now)
1. ✅ **Move analysis scripts** to `src/scripts/analysis/` (DONE)
2. **Create outputs directory structure**:
   ```bash
   mkdir -p outputs/{predictions,submissions,validation}
   mv predictions_*.csv outputs/predictions/
   mv submission_*.csv outputs/submissions/
   mv val_predictions_*.csv outputs/validation/
   ```
3. **Organize documentation**:
   ```bash
   mkdir -p docs/planning
   mv AGENTS.md Data_description.md docs/
   mv approach*.md todo.md docs/planning/
   mv src/appraoch1.md docs/planning/approach1.md  # Fix typo
   ```
4. **Create notebooks directory**:
   ```bash
   mkdir notebooks
   mv *.ipynb notebooks/
   ```

### Medium Priority (Next)
5. **Restructure src/** with subdirectories:
   - Move `evaluate_checkpoint.py`, `evaluate_validation.py`, `generate_dashboard_predictions.py` to `src/inference/`
   - Rename `data.py` → `src/data/dataset.py`
   - Rename `model.py` → `src/models/base.py`
   - Move `trainer.py` → `src/training/trainer.py`
   - Move `inference.py` → `src/inference/predictor.py`

6. **Create apps directory**:
   ```bash
   mkdir apps
   mv dashboard.py dashboard_prompt.txt apps/
   ```

### Low Priority (Later)
7. **Split models by approach**: Create separate files for each approach's architecture
8. **Add README files** in each major directory explaining contents
9. **Create requirements files**: Separate dev/prod dependencies if needed
10. **Add .gitignore entries** for outputs/, checkpoints/, logs/, wandb/

## Benefits

1. **Clear separation of concerns**: Data, models, training, inference, analysis
2. **Easier navigation**: Logical grouping of related files
3. **Better scalability**: Easy to add new approaches, models, analyses
4. **Cleaner root**: Only config files and README at root
5. **Improved discoverability**: Clear where to find/add code
6. **Better for collaboration**: Standard ML project structure

## Import Path Changes

After restructuring, update imports:
```python
# Old
from src.data import NFLDataset
from src.model import TrajectoryPredictor
from src.trainer import Trainer

# New
from src.data.dataset import NFLDataset
from src.models.base import TrajectoryPredictor
from src.training.trainer import Trainer
```

## Notes

- Keep backward compatibility during transition
- Update all import statements after moving files
- Run tests after each major restructuring step
- Update AGENTS.md with new structure
- Consider using symbolic links temporarily during transition
