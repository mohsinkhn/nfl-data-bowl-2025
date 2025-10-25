# Approach 1 Implementation Plan - Seq2Seq Baseline

## Overview
Implement a sequence-to-sequence model for NFL player trajectory prediction during pass plays. The model predicts player positions (x, y) after the ball is thrown, given pre-throw tracking data and ball landing location.

## Architecture Summary
- **Model Type**: Seq2Seq with encoder-decoder architecture
- **Backbone Options**: RNN (LSTM/GRU) or Conv1D for temporal encoding
- **Training**: Teacher forcing supported for decoder
- **Framework**: PyTorch Lightning for training orchestration

## Data Preparation Strategy

### Normalization Approach
1. **Centering**: Last pre-throw point at (0, 0) for each player
2. **Direction Normalization**: Rotate all plays so movement is left-to-right
3. **Field Normalization**: Scale (x, y) by field dimensions (120 yards × 53.3 yards)
4. **Angular Features**: Normalize orientation (o) and direction (dir) to relative angles

### Train/Val Split Strategy
- **Validation Fold 1**: Weeks 15-18 (4 weeks) - temporal validation
  - Training: Weeks 1-14 (14 weeks)
- **Validation Fold 2**: Weeks 9-12 (4 weeks) - mid-season validation to check temporal effects
  - Training: Weeks 1-8 + 13-18 (14 weeks, excluding val weeks to avoid leakage)

### Sequence Length
- Analyze pre-throw frame distribution
- Use 90th percentile as fixed encoder length
- Pad shorter sequences with zeros and use masking

## File Structure

```
src/
├── data.py              # NEW - Dataset class with normalization & collation (not using existing datasets.py)
├── model.py             # NEW - Seq2Seq models (not using existing models.py)
├── trainer.py           # NEW - Lightning module & training loop
├── inference.py         # NEW - Inference logic for test set
├── configs/             # NEW - Configuration files
│   ├── __init__.py
│   └── config_approach1_v0.py  # Default hyperparameters
├── utils/               # NEW - Utility functions
│   ├── __init__.py
│   ├── normalization.py # Coordinate transforms, rotation, scaling
│   └── metrics.py       # RMSE and other evaluation metrics
└── scripts/             # NEW - Data processing scripts
    ├── __init__.py
    ├── prepare_data.py  # Merge weeks, create train/val splits, save parquet
    ├── train.py         # Training script with config loading
    └── analyze_sequences.py # Compute sequence length statistics

tests/
├── test_data.py
├── test_model.py
└── test_normalization.py
```

## Implementation Phases

### Phase 1: Data Preparation Scripts ✓
**File**: `src/scripts/prepare_data.py`

**Responsibilities**:
- Load all input/output CSV files from `data/train/`
- Compute sequence statistics (encoder length distribution)
- Create train/val splits based on week numbers
- Apply basic preprocessing (sort by frame_id, create identifiers)
- Save processed data as parquet files:
  - `data/processed/fold1/train_input.parquet`
  - `data/processed/fold1/train_output.parquet`
  - `data/processed/fold1/val_input.parquet`
  - `data/processed/fold1/val_output.parquet`
  - `data/processed/fold2/train_input.parquet` (weeks 1-8,13-18)
  - `data/processed/fold2/train_output.parquet`
  - `data/processed/fold2/val_input.parquet` (weeks 9-12)
  - `data/processed/fold2/val_output.parquet`

**Arguments**:
```bash
--data_dir: Path to data folder (default: data/train)
--output_dir: Path to save processed files (default: data/processed)
--fold: Which fold to prepare ('fold1' or 'fold2' or 'both')
  - fold1: train=weeks 1-14, val=weeks 15-18
  - fold2: train=weeks 1-8,13-18, val=weeks 9-12
--max_encoder_len: Maximum encoder sequence length (default: None, auto-compute 90th percentile)
```

**File**: `src/scripts/analyze_sequences.py`

**Responsibilities**:
- Analyze pre-throw frame count distribution
- Compute percentile statistics for encoder length selection
- Generate visualizations of sequence length distributions
- Output recommendations for hyperparameters

---

### Phase 2: Utility Functions ✓
**File**: `src/utils/normalization.py`

**Functions**:
```python
def compute_last_frame_offset(player_data: np.ndarray) -> Tuple[float, float]
    """Extract (x, y) of last frame for centering"""

def center_coordinates(coords: np.ndarray, offset: Tuple[float, float]) -> np.ndarray
    """Center coordinates by subtracting offset"""

def compute_play_direction(player_data: np.ndarray, play_direction: str) -> float
    """Compute rotation angle to normalize play direction (left-to-right)"""

def rotate_coordinates(coords: np.ndarray, angle: float) -> np.ndarray
    """Rotate (x, y) coordinates by angle"""

def rotate_angles(angles: np.ndarray, angle: float) -> np.ndarray
    """Rotate orientation/direction angles"""

def normalize_by_field_size(coords: np.ndarray, field_dims: Tuple[float, float] = (120, 53.3)) -> np.ndarray
    """Normalize coordinates by field dimensions"""

def denormalize_predictions(preds: np.ndarray, offset: Tuple, angle: float, field_dims: Tuple) -> np.ndarray
    """Reverse all normalization transforms for final predictions"""

class CoordinateTransform:
    """Stateful transform that stores offset, angle, field_dims for reversibility"""
    def fit(self, player_data, play_direction): ...
    def transform(self, data): ...
    def inverse_transform(self, data): ...
```

**File**: `src/utils/metrics.py`

**Functions**:
```python
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float
    """Root Mean Squared Error"""

def per_player_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
    """RMSE per player for analysis"""

def per_frame_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
    """RMSE per frame to identify error accumulation"""
```

**File**: `src/configs/config_approach1_v0.py`

**Structure**:
```python
# Default configuration for Approach 1 Baseline
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class DataConfig:
    data_dir: str = "data/processed/fold1"
    batch_size: int = 32
    max_encoder_len: int = None  # Auto-computed from data
    max_decoder_len: int = 50
    feature_cols: List[str] = None  # Default: ['x', 'y', 's', 'a', 'dir', 'o']
    num_workers: int = 4
    normalize: bool = True
    rotation_normalize: bool = True
    
    def __post_init__(self):
        if self.feature_cols is None:
            self.feature_cols = ['x', 'y', 's', 'a', 'dir', 'o']

@dataclass
class ModelConfig:
    input_dim: int = 6  # x, y, s, a, dir, o
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    encoder_type: str = 'lstm'  # 'lstm' or 'gru'
    decoder_type: str = 'lstm'  # 'lstm' or 'gru'
    output_dim: int = 2  # (x, y)
    teacher_forcing_ratio: float = 0.5

@dataclass
class TrainingConfig:
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = 'reduce_on_plateau'
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints/approach1_v0"
    wandb_project: str = "nfl-data-bowl-2025"
    wandb_entity: str = None  # Optional team/org slug
    wandb_run_name: str = None
    wandb_tags: List[str] = None
    wandb_mode: str = "online"  # "online", "offline", or "disabled"
    
def get_default_config():
    """Returns default configuration objects"""
    return DataConfig(), ModelConfig(), TrainingConfig()
```

---

### Phase 3: New Dataset ✓
**File**: `src/data.py` (new file, not using existing datasets.py)

**Implementation**:
```python
class NFLTrajectoryDataset(Dataset):
    """
    Enhanced dataset with normalization and padding
    
    Args:
        input_parquet: Path to input parquet file
        output_parquet: Path to output parquet file
        max_encoder_len: Fixed encoder sequence length
        max_decoder_len: Maximum decoder sequence length
        feature_cols: List of feature column names
        normalize: Whether to apply normalization
        rotation_normalize: Whether to normalize play direction
    """
    
    def __init__(self, input_parquet, output_parquet, max_encoder_len, 
                 max_decoder_len, feature_cols, normalize=True, rotation_normalize=True):
        # Load parquet files
        # Store transforms per sample
        # Precompute normalization parameters
        
    def __getitem__(self, idx):
        # Load sample
        # Apply normalization
        # Pad sequences
        # Return:
        #   encoder_input: (max_encoder_len, feature_dim)
        #   decoder_input: (max_decoder_len, feature_dim)
        #   decoder_target: (max_decoder_len, output_dim)
        #   encoder_mask: (max_encoder_len,)
        #   decoder_mask: (max_decoder_len,)
        #   metadata: dict with game_id, play_id, nfl_id, transform params

def collate_fn(batch):
    """Collate function for DataLoader"""
    # Stack tensors
    # Return batch dict
```

---

### Phase 4: Seq2Seq Models ✓
**File**: `src/model.py` (new file, not using existing models.py)

**Classes**:
```python
class LSTMEncoder(nn.Module):
    """LSTM-based encoder for trajectory sequences"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        # LSTM layers
        # Output: hidden states (all time steps) + final hidden state
        
    def forward(self, x, mask=None):
        # Pack padded sequences
        # LSTM forward
        # Unpack
        # Return encoder_outputs, (h_n, c_n)

class LSTMDecoder(nn.Module):
    """LSTM-based decoder with teacher forcing support"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        # LSTM layers
        # Output projection to (x, y)
        
    def forward(self, x, hidden, encoder_outputs=None, teacher_forcing_ratio=0.5):
        # Support autoregressive generation
        # Optionally attend to encoder outputs (basic attention)
        # Return predictions: (batch, seq_len, output_dim)

class GRUEncoder(nn.Module):
    """GRU-based encoder for trajectory sequences"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        # GRU layers
        # Output: hidden states (all time steps) + final hidden state
        
    def forward(self, x, mask=None):
        # Pack padded sequences
        # GRU forward
        # Unpack
        # Return encoder_outputs, h_n

class GRUDecoder(nn.Module):
    """GRU-based decoder with teacher forcing support"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        # GRU layers
        # Output projection to (x, y)
        
    def forward(self, x, hidden, encoder_outputs=None, teacher_forcing_ratio=0.5):
        # Support autoregressive generation
        # Optionally attend to encoder outputs (basic attention)
        # Return predictions: (batch, seq_len, output_dim)

class Seq2SeqTrajectoryModel(nn.Module):
    """Complete Seq2Seq model combining encoder and decoder"""
    def __init__(self, encoder, decoder, device):
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, encoder_input, decoder_input, teacher_forcing_ratio=0.5):
        # Encode
        # Decode with optional teacher forcing
        # Return predictions

    def predict(self, encoder_input, max_len):
        # Inference mode (no teacher forcing)
        # Autoregressive generation
        # Return predictions
```

**Note**: Conv1D encoder dropped from initial baseline. Focus on LSTM/GRU only.

---

### Phase 5: Lightning Trainer ✓
**File**: `src/trainer.py`

**Class**:
```python
class TrajectoryPredictionModule(pl.LightningModule):
    """PyTorch Lightning module for trajectory prediction"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig, 
                 data_config: DataConfig, transform_params: dict = None):
        super().__init__()
        self.save_hyperparameters()
        
        # Store configs for checkpoint saving
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.transform_params = transform_params  # For denormalization
        
        # Build encoder
        # Build decoder
        # Build Seq2Seq model
        # Loss function (MSE)
        
    def forward(self, encoder_input, decoder_input, teacher_forcing_ratio):
        return self.model(encoder_input, decoder_input, teacher_forcing_ratio)
    
    def training_step(self, batch, batch_idx):
        # Unpack batch
        # Forward pass with teacher forcing
        # Compute loss (MSE on x, y)
        # Log metrics
        # Return loss
        
    def validation_step(self, batch, batch_idx):
        # Forward pass without teacher forcing (inference mode)
        # Compute RMSE
        # Log metrics
        
    def configure_optimizers(self):
        # Adam optimizer
        # Optional: ReduceLROnPlateau or CosineAnnealing scheduler
        
    def predict_step(self, batch, batch_idx):
        # Inference for test set
        # Denormalize predictions using self.transform_params
        # Return predictions with metadata
        
    def on_save_checkpoint(self, checkpoint):
        # Save configs and transform params with checkpoint
        checkpoint['model_config'] = asdict(self.model_config)
        checkpoint['training_config'] = asdict(self.training_config)
        checkpoint['data_config'] = asdict(self.data_config)
        checkpoint['transform_params'] = self.transform_params
        
    def on_load_checkpoint(self, checkpoint):
        # Restore configs and transform params
        self.model_config = ModelConfig(**checkpoint['model_config'])
        self.training_config = TrainingConfig(**checkpoint['training_config'])
        self.data_config = DataConfig(**checkpoint['data_config'])
        self.transform_params = checkpoint['transform_params']

def train_model(model_config: ModelConfig, data_config: DataConfig, 
                training_config: TrainingConfig):
    """
    Main training function
    
    Args:
        model_config: Model hyperparameters
        data_config: Data loading config
        training_config: Training hyperparameters
    """
    # Create datasets
    train_dataset = NFLTrajectoryDataset(...)
    val_dataset = NFLTrajectoryDataset(...)
    
    # Extract transform params from train dataset (for saving with checkpoint)
    transform_params = train_dataset.get_transform_params()
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=..., collate_fn=...)
    val_loader = DataLoader(val_dataset, batch_size=..., collate_fn=...)
    
    # Create Lightning module with transform params
    model = TrajectoryPredictionModule(model_config, training_config, data_config, transform_params)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(...)
    early_stopping = EarlyStopping(...)
    lr_monitor = LearningRateMonitor(...)

    # Logging
    wandb_logger = WandbLogger(
        project=training_config.wandb_project,
        entity=training_config.wandb_entity,
        name=training_config.wandb_run_name,
        tags=training_config.wandb_tags,
        mode=training_config.wandb_mode,
    )
    wandb_logger.watch(model, log="all", log_freq=500)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        gradient_clip_val=training_config.gradient_clip_val,
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return trainer, model
```

---

### Phase 6: Inference Pipeline ✓
**File**: `src/inference.py`

**Functions**:
```python
def load_test_data(test_input_path: str, test_target_path: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load test input and target structure"""
    
def create_test_dataset(test_input_df, test_target_df, model_config, data_config):
    """Create dataset for test set"""
    
def run_inference(checkpoint_path: str, test_dataset, output_path: str):
    """
    Run inference on test set and generate submission file
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        test_dataset: Test dataset
        output_path: Path to save submission CSV
    """
    # Load model from checkpoint (configs and transform_params are restored automatically)
    model = TrajectoryPredictionModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Transform params are available in model.transform_params for denormalization
    
    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=..., collate_fn=...)
    
    # Predict
    trainer = pl.Trainer(accelerator='auto', devices=1)
    predictions = trainer.predict(model, test_loader)
    
    # Denormalize predictions
    # Format as submission: id, x, y
    # Save to CSV
    
def create_submission_file(predictions: List[Dict], output_path: str):
    """Format predictions as submission file"""
    # Create DataFrame with columns: id, x, y
    # id format: {game_id}_{play_id}_{nfl_id}_{frame_id}
    # Save CSV
```

---

### Phase 7: Main Training Script ✓
**File**: `src/scripts/train.py`

```python
import argparse
from pathlib import Path
from dataclasses import replace, asdict
import importlib.util
from src.configs.config_approach1_v0 import DataConfig, ModelConfig, TrainingConfig, get_default_config
from src.trainer import train_model

def load_config_from_file(config_path: str):
    """Load configuration from Python file"""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.get_default_config()

def parse_args():
    parser = argparse.ArgumentParser(description='Train trajectory prediction model')
    
    # Config file
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to config file (e.g., src/configs/config_approach1_v0.py)')
    parser.add_argument('--fold', type=str, default='fold1', choices=['fold1', 'fold2'],
                       help='Which fold to use for training')
    
    # Data arguments (overrides)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--max_encoder_len', type=int, default=None)
    parser.add_argument('--max_decoder_len', type=int, default=None)
    
    # Model arguments (overrides)
    parser.add_argument('--encoder_type', type=str, default=None, choices=['lstm', 'gru'])
    parser.add_argument('--decoder_type', type=str, default=None, choices=['lstm', 'gru'])
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=None)
    
    # Training arguments (overrides)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--gradient_clip_val', type=float, default=None)
    parser.add_argument('--early_stopping_patience', type=int, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_tags', type=str, nargs="+", default=None)
    parser.add_argument('--wandb_mode', type=str, default=None, choices=['online', 'offline', 'disabled'])
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load default config or from file
    if args.config:
        data_config, model_config, training_config = load_config_from_file(args.config)
    else:
        data_config, model_config, training_config = get_default_config()
    
    # Set fold-specific data directory
    if args.data_dir is None:
        data_config.data_dir = f"data/processed/{args.fold}"
    
    # Apply overrides to data_config
    override_dict = {}
    if args.data_dir: override_dict['data_dir'] = args.data_dir
    if args.batch_size: override_dict['batch_size'] = args.batch_size
    if args.num_workers: override_dict['num_workers'] = args.num_workers
    if args.max_encoder_len: override_dict['max_encoder_len'] = args.max_encoder_len
    if args.max_decoder_len: override_dict['max_decoder_len'] = args.max_decoder_len
    if override_dict:
        data_config = replace(data_config, **override_dict)
    
    # Apply overrides to model_config
    override_dict = {}
    if args.encoder_type: override_dict['encoder_type'] = args.encoder_type
    if args.decoder_type: override_dict['decoder_type'] = args.decoder_type
    if args.hidden_dim: override_dict['hidden_dim'] = args.hidden_dim
    if args.num_layers: override_dict['num_layers'] = args.num_layers
    if args.dropout: override_dict['dropout'] = args.dropout
    if args.teacher_forcing_ratio: override_dict['teacher_forcing_ratio'] = args.teacher_forcing_ratio
    if override_dict:
        model_config = replace(model_config, **override_dict)
    
    # Apply overrides to training_config
    override_dict = {}
    if args.max_epochs: override_dict['max_epochs'] = args.max_epochs
    if args.learning_rate: override_dict['learning_rate'] = args.learning_rate
    if args.weight_decay: override_dict['weight_decay'] = args.weight_decay
    if args.gradient_clip_val: override_dict['gradient_clip_val'] = args.gradient_clip_val
    if args.early_stopping_patience: override_dict['early_stopping_patience'] = args.early_stopping_patience
    if args.checkpoint_dir: override_dict['checkpoint_dir'] = args.checkpoint_dir
    if args.wandb_project: override_dict['wandb_project'] = args.wandb_project
    if args.wandb_entity: override_dict['wandb_entity'] = args.wandb_entity
    if args.wandb_run_name: override_dict['wandb_run_name'] = args.wandb_run_name
    if args.wandb_tags is not None: override_dict['wandb_tags'] = args.wandb_tags
    if args.wandb_mode: override_dict['wandb_mode'] = args.wandb_mode
    if override_dict:
        training_config = replace(training_config, **override_dict)
    
    # Print final configs
    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(f"Data Config: {asdict(data_config)}")
    print(f"Model Config: {asdict(model_config)}")
    print(f"Training Config: {asdict(training_config)}")
    print("=" * 80)
    
    # Train
    trainer, model = train_model(model_config, data_config, training_config)
    
    print(f"\nTraining complete! Best model saved to {training_config.checkpoint_dir}")

if __name__ == '__main__':
    main()
```

---

### Phase 8: Main Inference Script ✓
**File**: `src/inference.py` (in src/ root, not scripts/)

```python
import argparse
from src.inference import load_test_data, create_test_dataset, run_inference
from src.trainer import TrajectoryPredictionModule

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_input', type=str, default='data/test_input.csv')
    parser.add_argument('--test_target', type=str, default='data/test.csv')
    parser.add_argument('--output_path', type=str, default='submission.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load checkpoint to extract configs
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    data_config = DataConfig(**checkpoint['data_config'])
    
    # Load test data
    test_input_df, test_target_df = load_test_data(args.test_input, args.test_target)
    
    # Create test dataset (configs loaded from checkpoint)
    test_dataset = create_test_dataset(test_input_df, test_target_df, data_config)
    
    # Run inference (model loads configs and transform_params internally)
    run_inference(args.checkpoint_path, test_dataset, args.output_path)
    
    print(f"Submission saved to {args.output_path}")

if __name__ == '__main__':
    main()
```

---

## Testing Strategy

### Unit Tests
```python
# tests/test_normalization.py
- test_center_coordinates()
- test_rotate_coordinates()
- test_normalize_by_field_size()
- test_denormalize_predictions()
- test_coordinate_transform_invertibility()

# tests/test_data.py
- test_dataset_loading()
- test_sequence_padding()
- test_collate_fn()
- test_normalization_in_dataset()

# tests/test_model.py
- test_lstm_encoder_forward()
- test_gru_encoder_forward()
- test_lstm_decoder_forward()
- test_gru_decoder_forward()
- test_seq2seq_forward()
- test_seq2seq_predict()
- test_model_output_shapes()

# tests/test_metrics.py
- test_rmse()
- test_per_player_rmse()
```

---

## Dependencies to Add

Update `pyproject.toml`:
```toml
dependencies = [
    # ... existing
    "pytorch-lightning>=2.0.0",
    "wandb>=0.17.0",
    "pyarrow>=14.0.0",  # for parquet
]
```

---

## Execution Order

1. **Data Preparation**:
   ```bash
   # Analyze sequences
   python src/scripts/analyze_sequences.py --data_dir data/train
   
   # Prepare fold1 (train: weeks 1-14, val: weeks 15-18)
   python src/scripts/prepare_data.py --data_dir data/train --output_dir data/processed --fold fold1
   
   # Prepare fold2 (train: weeks 1-8,13-18, val: weeks 9-12)
   python src/scripts/prepare_data.py --data_dir data/train --output_dir data/processed --fold fold2
   ```

2. **Training**:
   ```bash
   # Train on fold1 with default config
   python src/scripts/train.py --fold fold1
   
   # Train on fold1 with config file and overrides
   python src/scripts/train.py \
       --config src/configs/config_approach1_v0.py \
       --fold fold1 \
       --encoder_type lstm \
       --hidden_dim 128 \
       --batch_size 32 \
       --max_epochs 50
       
   # Train on fold2
   python src/scripts/train.py --fold fold2 --hidden_dim 256
   ```

3. **Inference**:
   ```bash
   # Configs and transform params loaded automatically from checkpoint
   python -m src.inference \
       --checkpoint_path checkpoints/approach1_v0/best_model.ckpt \
       --test_input data/test_input.csv \
       --test_target data/test.csv \
       --output_path submission.csv
   ```

---

## Key Design Decisions

### 1. Normalization Strategy
- **Last frame centering**: Makes model focus on relative movements
- **Play direction normalization**: Reduces variance, model learns canonical direction
- **Field normalization**: Helps with gradient flow and generalization
- **Reversibility**: Store transform parameters for each sample to denormalize predictions

### 2. Sequence Handling
- **Fixed encoder length**: Simplifies batching, uses 90th percentile
- **Padding + Masking**: Handle variable lengths cleanly
- **Decoder length**: Variable per sample (from `num_frames_output`)

### 3. Model Architecture
- **Seq2Seq**: Natural fit for sequence continuation task
- **Teacher forcing**: Helps training stability, gradually reduce ratio
- **Encoder options**: LSTM (default) or GRU (Conv1D dropped from baseline)

### 4. Training Strategy
- **Lightning**: Clean abstraction, easy logging, checkpointing
- **Early stopping**: Prevent overfitting
- **Gradient clipping**: Stabilize RNN training
- **Learning rate scheduling**: Improve convergence

### 5. Evaluation Strategy
- **Two validation folds**: 
  - Fold 1: Temporal validation (train: weeks 1-14, val: weeks 15-18)
  - Fold 2: Mid-season validation (train: weeks 1-8,13-18, val: weeks 9-12) - no leakage
- **Per-frame RMSE**: Identify error accumulation over time
- **Per-player RMSE**: Analyze performance by player role

### 6. Checkpoint Management
- **Save with checkpoint**: Model configs, data configs, training configs, transform parameters
- **Load from checkpoint**: All configs and transform params restored automatically
- **Enables reproducible inference**: No need to pass configs separately

---

## Future Directions

1. **Input embeddings**: Project heterogeneous inputs via `Linear → BatchNorm → GELU` before the encoder (cf. LSTNet – Lai et al., 2018; Informer – Zhou et al., 2021). Added in code; tune projection dim/dropout.
2. **Expanded context**: One-hot player role + normalised height/weight now travel through the pipeline; ablate with `--no-player-role/--no-player-attributes`.
3. **Polar targets (Approach 2)**: Reparameterise decoder outputs as `(Δt, Δθ)` relative to the last encoder heading, integrate to recover `(x, y)`. Literature: Social-LSTM/Social-GAN (Alahi et al., CVPR 2016; Gupta et al., CVPR 2018), driving-trajectory embeddings (Li et al., ICRA 2020), sports trajectory models (Lu et al., KDD 2019). See `approach2.md` for plan.
4. **Transformer/attention**: Swap recurrent blocks for sparse attention (e.g., Informer) once input projection is stable; optional cross-attention on encoder states.
5. **Interaction modelling**: Graph or social pooling layers to capture defender–receiver coupling.
6. **Ensembling**: Blend Cartesian and polar-trained models for competition submissions.

---

## Success Metrics

- **Primary**: RMSE on validation set < baseline threshold (TBD after EDA)
- **Secondary**: 
  - Error doesn't explode over longer sequences
  - Model generalizes across validation folds
  - Inference time < 1 second per play
  - Submission file format correct

---

## Timeline Estimate

- **Phase 1-2** (Utils & Scripts): 1-2 days
- **Phase 3-4** (Dataset & Models): 2-3 days
- **Phase 5-6** (Training & Inference): 2-3 days
- **Phase 7-8** (Integration & Testing): 1-2 days
- **Total**: ~1-1.5 weeks for complete baseline

---

## Open Questions / Design Choices

1. **Encoder length**: Confirm 90th percentile is appropriate after sequence analysis
2. **Feature selection**: Start with (x, y, s, a, dir, o) - full set
3. **Ball landing location**: Include as static context or separate input channel?
4. **Player role encoding**: Add as categorical feature or separate models?
5. **Attention**: Basic attention or full transformer-style? (Future extension)
6. **Loss function**: MSE (default) vs MAE vs Huber?
7. **Decoder initialization**: Use last encoder state or learned initialization?

## Summary of Key Changes from Original Plan

1. ✅ **Fixed data leakage in fold2**: Train on weeks 1-8,13-18 (excluding val weeks 9-12)
2. ✅ **Not using existing datasets.py/models.py**: Created new src/data.py and src/model.py
3. ✅ **Config file support**: Added src/configs/config_approach1_v0.py with override capability
4. ✅ **Checkpoint enhancements**: Save configs and transform_params for reproducible inference
5. ✅ **Dropped Conv1D**: Focus on LSTM/GRU encoders only for baseline
6. ✅ **Fold-specific data dirs**: data/processed/fold1/ and data/processed/fold2/

---

## Notes

- Keep all paths configurable via arguments (no hardcoding)
- Follow black formatting (88 chars)
- Add docstrings to all public functions
- Log hyperparameters and metrics to wandb
- Save model config with checkpoint for reproducible inference
- Version control processed data directory structure
