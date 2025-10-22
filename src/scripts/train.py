"""Command-line entry point for training the seq2seq trajectory model."""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import asdict, replace
from pathlib import Path
from pprint import pprint
from typing import Tuple

from pytorch_lightning.callbacks import ModelCheckpoint

from src.configs.config_approach1_v0 import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    get_default_config,
)
from src.trainer import train_model


def _load_config_module(config_path: Path) -> Tuple[DataConfig, ModelConfig, TrainingConfig]:
    spec = importlib.util.spec_from_file_location("approach_config", config_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:  # pragma: no cover - handled via argparse type checking
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
    valid_updates = {}
    for key, value in updates.items():
        if value is None:
            continue
        if not hasattr(config, key):
            continue
        valid_updates[key] = value
    if not valid_updates:
        return config
    return replace(config, **valid_updates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the seq2seq trajectory prediction baseline",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a Python config file exporting get_default_config()",
    )
    parser.add_argument(
        "--fold",
        type=str,
        choices=["fold1", "fold2"],
        default=None,
        help="Training fold to use (maps to data/processed/{fold})",
    )

    data_group = parser.add_argument_group("Data overrides")
    data_group.add_argument("--data-dir", type=Path, default=None)
    data_group.add_argument("--batch-size", type=int, default=None)
    data_group.add_argument("--num-workers", type=int, default=None)
    data_group.add_argument("--max-encoder-len", type=int, default=None)
    data_group.add_argument("--max-decoder-len", type=int, default=None)
    data_group.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable feature normalization",
    )
    data_group.add_argument(
        "--no-rotation-normalize",
        action="store_true",
        help="Disable play-direction rotation normalization",
    )

    model_group = parser.add_argument_group("Model overrides")
    model_group.add_argument("--encoder-type", choices=["lstm", "gru"], default=None)
    model_group.add_argument("--decoder-type", choices=["lstm", "gru"], default=None)
    model_group.add_argument("--hidden-dim", type=int, default=None)
    model_group.add_argument("--num-layers", type=int, default=None)
    model_group.add_argument("--dropout", type=float, default=None)
    model_group.add_argument("--teacher-forcing-ratio", type=float, default=None)
    model_group.add_argument(
        "--use-attention",
        action="store_true",
        help="Enable decoder attention (experimental)",
    )
    model_group.add_argument(
        "--use-ball-context",
        action="store_true",
        help="Append ball landing context to decoder inputs",
    )
    model_group.add_argument(
        "--bidirectional",
        action="store_true",
        help="Enable bidirectional encoder (not supported yet)",
    )

    train_group = parser.add_argument_group("Training overrides")
    train_group.add_argument("--max-epochs", type=int, default=None)
    train_group.add_argument("--learning-rate", type=float, default=None)
    train_group.add_argument("--weight-decay", type=float, default=None)
    train_group.add_argument("--gradient-clip-val", type=float, default=None)
    train_group.add_argument(
        "--scheduler",
        type=str,
        choices=["reduce_on_plateau", "cosine", "none"],
        default=None,
    )
    train_group.add_argument("--scheduler-patience", type=int, default=None)
    train_group.add_argument("--scheduler-factor", type=float, default=None)
    train_group.add_argument("--scheduler-min-lr", type=float, default=None)
    train_group.add_argument("--early-stopping-patience", type=int, default=None)
    train_group.add_argument("--monitor-metric", type=str, default=None)
    train_group.add_argument("--monitor-mode", choices=["min", "max"], default=None)
    train_group.add_argument("--save-top-k", type=int, default=None)
    train_group.add_argument("--checkpoint-dir", type=Path, default=None)
    train_group.add_argument("--log-dir", type=Path, default=None)
    train_group.add_argument("--log-every-n-steps", type=int, default=None)
    train_group.add_argument("--val-check-interval", type=float, default=None)
    train_group.add_argument("--accelerator", type=str, default=None)
    train_group.add_argument("--devices", type=int, default=None)
    train_group.add_argument("--precision", type=str, default=None)
    train_group.add_argument("--seed", type=int, default=None)
    train_group.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
    )
    train_group.add_argument("--wandb-project", type=str, default=None)
    train_group.add_argument("--wandb-entity", type=str, default=None)
    train_group.add_argument("--wandb-run-name", type=str, default=None)
    train_group.add_argument("--wandb-tags", nargs="*", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config is not None:
        data_config, model_config, training_config = _load_config_module(args.config)
    else:
        data_config, model_config, training_config = get_default_config()

    if args.bidirectional:
        raise NotImplementedError("Bidirectional encoders are not supported in this baseline.")

    if args.fold is not None and args.data_dir is None:
        base_dir = Path(data_config.data_dir).expanduser()
        root_dir = base_dir.parent if base_dir.name.startswith("fold") else base_dir
        data_config = replace(data_config, data_dir=str(root_dir / args.fold))

    data_overrides = {
        "data_dir": str(args.data_dir) if args.data_dir is not None else None,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_encoder_len": args.max_encoder_len,
        "max_decoder_len": args.max_decoder_len,
    }
    if args.no_normalize:
        data_overrides["normalize"] = False
    if args.no_rotation_normalize:
        data_overrides["rotation_normalize"] = False
    data_config = _update_dataclass(data_config, data_overrides)

    model_overrides = {
        "encoder_type": args.encoder_type,
        "decoder_type": args.decoder_type,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "teacher_forcing_ratio": args.teacher_forcing_ratio,
        "use_attention": True if args.use_attention else None,
        "use_ball_context": True if args.use_ball_context else None,
    }
    model_config = _update_dataclass(model_config, model_overrides)

    training_overrides = {
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_val": args.gradient_clip_val,
        "scheduler": args.scheduler,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_min_lr": args.scheduler_min_lr,
        "early_stopping_patience": args.early_stopping_patience,
        "monitor_metric": args.monitor_metric,
        "monitor_mode": args.monitor_mode,
        "save_top_k": args.save_top_k,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "log_dir": str(args.log_dir) if args.log_dir else None,
        "log_every_n_steps": args.log_every_n_steps,
        "val_check_interval": args.val_check_interval,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": args.precision,
        "seed": args.seed,
    }
    training_config = _update_dataclass(training_config, training_overrides)

    if args.wandb_mode is not None:
        setattr(training_config, "wandb_mode", args.wandb_mode)
    if args.wandb_project is not None:
        setattr(training_config, "wandb_project", args.wandb_project)
    if args.wandb_entity is not None:
        setattr(training_config, "wandb_entity", args.wandb_entity)
    if args.wandb_run_name is not None:
        setattr(training_config, "wandb_run_name", args.wandb_run_name)
    if args.wandb_tags is not None:
        setattr(training_config, "wandb_tags", args.wandb_tags)

    print("\n=== Training Configuration ===")
    print("DataConfig:")
    pprint(asdict(data_config))
    print("\nModelConfig:")
    pprint(asdict(model_config))
    print("\nTrainingConfig:")
    pprint(asdict(training_config))

    trainer, _ = train_model(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
    )

    best_checkpoint = None
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_checkpoint = callback.best_model_path
            break

    if best_checkpoint:
        print(f"\nBest checkpoint saved to: {best_checkpoint}")
    else:
        print("\nTraining completed without checkpoint information.")

    print("\nFinished training run.")
    print(f"Last logged metrics: {trainer.callback_metrics}")


if __name__ == "__main__":
    main()
