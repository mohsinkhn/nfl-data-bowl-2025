"""Pretrain the encoder on a next-frame prediction objective."""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import asdict, replace
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.configs.config_approach1_v0 import DataConfig, ModelConfig, TrainingConfig, get_default_config
from src.trainer import (
    EncoderNextFrameModule,
    _build_dataloaders,
    _build_logger,
)


def _load_config_module(config_path: Path) -> Tuple[DataConfig, ModelConfig, TrainingConfig]:
    spec = importlib.util.spec_from_file_location("pretrain_config", config_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Cannot load configuration module from {config_path}")
    spec.loader.exec_module(module)
    if not hasattr(module, "get_default_config"):
        raise AttributeError(
            f"Configuration file {config_path} must define get_default_config()"
        )
    configs = module.get_default_config()
    if len(configs) != 3:
        raise ValueError(
            "get_default_config() must return (DataConfig, ModelConfig, TrainingConfig)"
        )
    return configs  # type: ignore[return-value]


def _update_dataclass(config, updates):
    valid = {}
    for key, value in updates.items():
        if value is None:
            continue
        if not hasattr(config, key):
            continue
        valid[key] = value
    if not valid:
        return config
    return replace(config, **valid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain encoder on next-frame prediction")
    parser.add_argument("--config", type=Path, default=None, help="Optional config file path")
    parser.add_argument("--fold", type=str, choices=["fold1", "fold2"], default=None)
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pretrain_next_frame"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config is not None:
        data_config, model_config, training_config = _load_config_module(args.config)
    else:
        data_config, model_config, training_config = get_default_config()

    if args.fold is not None and args.data_dir is None:
        base_dir = Path(data_config.data_dir).expanduser()
        root_dir = base_dir.parent if base_dir.name.startswith("fold") else base_dir
        data_config = replace(data_config, data_dir=str(root_dir / args.fold))

    data_overrides = {
        "data_dir": str(args.data_dir) if args.data_dir is not None else None,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    data_config = _update_dataclass(data_config, data_overrides)

    training_overrides = {
        "max_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "checkpoint_dir": str(args.output_dir),
    }
    training_config = _update_dataclass(training_config, training_overrides)

    train_loader, val_loader, _ = _build_dataloaders(data_config)

    train_dataset = train_loader.dataset
    if hasattr(train_dataset, "input_dim"):
        model_config = replace(model_config, input_dim=train_dataset.input_dim)

    module = EncoderNextFrameModule(model_config=model_config, training_config=training_config)

    Path(training_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=training_config.checkpoint_dir,
            monitor=training_config.monitor_metric,
            mode=training_config.monitor_mode,
            save_top_k=training_config.save_top_k,
            filename="pretrain-{epoch:02d}-{val_rmse:.3f}",
        ),
        EarlyStopping(
            monitor=training_config.monitor_metric,
            mode=training_config.monitor_mode,
            patience=training_config.early_stopping_patience,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = _build_logger(training_config)

    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        accelerator=training_config.accelerator,
        devices=training_config.devices,
        precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=training_config.log_every_n_steps,
        val_check_interval=training_config.val_check_interval,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
