"""PyTorch Lightning training utilities for the seq2seq trajectory baseline."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

try:  # Optional dependency; fall back to CSV logs when unavailable.
    from pytorch_lightning.loggers import WandbLogger
except ImportError:  # pragma: no cover - wandb might be absent in CI.
    WandbLogger = None  # type: ignore[misc,assignment]

from src.configs.config_approach1_v0 import DataConfig, ModelConfig, TrainingConfig
from src.model import (
    GRUDecoder,
    GRUEncoder,
    LSTMDecoder,
    LSTMEncoder,
    Seq2SeqTrajectoryModel,
)


def _build_encoder(config: ModelConfig) -> nn.Module:
    """Instantiate the encoder specified by the model configuration."""
    if config.bidirectional:
        raise NotImplementedError(
            "Bidirectional encoders are not implemented for Approach 1"
        )

    if config.encoder_type == "lstm":
        return LSTMEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    if config.encoder_type == "gru":
        return GRUEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported encoder type: {config.encoder_type}")


def _build_decoder(config: ModelConfig) -> nn.Module:
    """Instantiate the decoder specified by the model configuration."""
    if config.decoder_type == "lstm":
        return LSTMDecoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )
    if config.decoder_type == "gru":
        return GRUDecoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )
    raise ValueError(f"Unsupported decoder type: {config.decoder_type}")


class TrajectoryPredictionModule(pl.LightningModule):
    """Lightning module wrapping the seq2seq trajectory predictor."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        transform_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "model_config": asdict(model_config),
                "training_config": asdict(training_config),
                "data_config": asdict(data_config),
            }
        )

        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.transform_params = transform_params

        self.encoder = _build_encoder(model_config)
        self.decoder = _build_decoder(model_config)
        self.model = Seq2SeqTrajectoryModel(self.encoder, self.decoder)

        self.criterion = nn.MSELoss(reduction="none")

    def forward(
        self,
        encoder_inputs: Tensor,
        decoder_inputs: Tensor,
        teacher_forcing_ratio: float,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.model(
            encoder_inputs,
            decoder_inputs,
            teacher_forcing_ratio=teacher_forcing_ratio,
            encoder_mask=encoder_mask,
        )

    def _masked_mse(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        losses = self.criterion(predictions, targets)
        if mask is not None:
            mask = mask.to(losses.dtype)
            expanded_mask = mask.unsqueeze(-1)
            losses = losses * expanded_mask
            valid_steps = expanded_mask.sum()
            normalizer = valid_steps * predictions.size(-1)
        else:
            normalizer = torch.tensor(
                losses.numel(), device=losses.device, dtype=losses.dtype
            )
        normalizer = torch.clamp(normalizer, min=1.0)
        return losses.sum() / normalizer

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: D401
        encoder_inputs = batch["encoder_input"]
        decoder_inputs = batch["decoder_input"]
        decoder_targets = batch["decoder_target"]
        encoder_mask = batch.get("encoder_mask")
        decoder_mask = batch.get("decoder_mask")

        predictions = self(
            encoder_inputs,
            decoder_inputs,
            teacher_forcing_ratio=self.model_config.teacher_forcing_ratio,
            encoder_mask=encoder_mask,
        )
        loss = self._masked_mse(predictions, decoder_targets, decoder_mask)
        rmse = torch.sqrt(torch.clamp(loss, min=0.0))

        batch_size = encoder_inputs.size(0)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_rmse",
            rmse,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        encoder_inputs = batch["encoder_input"]
        decoder_inputs = batch["decoder_input"]
        decoder_targets = batch["decoder_target"]
        encoder_mask = batch.get("encoder_mask")
        decoder_mask = batch.get("decoder_mask")

        predictions = self(
            encoder_inputs,
            decoder_inputs,
            teacher_forcing_ratio=0.0,
            encoder_mask=encoder_mask,
        )
        loss = self._masked_mse(predictions, decoder_targets, decoder_mask)
        rmse = torch.sqrt(torch.clamp(loss, min=0.0))

        batch_size = encoder_inputs.size(0)
        self.log(
            "val_loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            self.training_config.monitor_metric,
            rmse,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def predict_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        encoder_inputs = batch["encoder_input"]
        decoder_inputs = batch["decoder_input"]
        encoder_mask = batch.get("encoder_mask")
        decoder_mask = batch.get("decoder_mask")
        metadata = batch.get("metadata")

        start_token = decoder_inputs[:, 0]
        steps = decoder_inputs.size(1)
        predictions = self.model.predict(
            encoder_inputs,
            decoder_start=start_token,
            prediction_steps=steps,
            encoder_mask=encoder_mask,
        )

        return {
            "predictions": predictions,
            "decoder_mask": decoder_mask,
            "metadata": metadata,
            "transform_params": self.transform_params,
        }

    def configure_optimizers(self):  # noqa: D401
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        if self.training_config.scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.training_config.monitor_mode,
                factor=self.training_config.scheduler_factor,
                patience=self.training_config.scheduler_patience,
                min_lr=self.training_config.scheduler_min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.training_config.monitor_metric,
                },
            }

        if self.training_config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.training_config.max_epochs,
            )
            return [optimizer], [scheduler]

        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["model_config"] = asdict(self.model_config)
        checkpoint["training_config"] = asdict(self.training_config)
        checkpoint["data_config"] = asdict(self.data_config)
        checkpoint["transform_params"] = self.transform_params

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.model_config = ModelConfig(**checkpoint["model_config"])
        self.training_config = TrainingConfig(**checkpoint["training_config"])
        self.data_config = DataConfig(**checkpoint["data_config"])
        self.transform_params = checkpoint.get("transform_params")


def _build_logger(training_config: TrainingConfig):
    """Create the logger requested by the training configuration."""
    wandb_mode = getattr(training_config, "wandb_mode", "disabled")
    if WandbLogger is not None and wandb_mode != "disabled":
        logger = WandbLogger(
            project=getattr(training_config, "wandb_project", None),
            entity=getattr(training_config, "wandb_entity", None),
            name=getattr(training_config, "wandb_run_name", None),
            tags=getattr(training_config, "wandb_tags", None),
            mode=wandb_mode,
        )
        return logger

    os.makedirs(training_config.log_dir, exist_ok=True)
    return CSVLogger(save_dir=training_config.log_dir, name="lightning")


def _build_dataloaders(
    data_config: DataConfig,
):
    """Create training and validation dataloaders based on the configuration."""
    try:
        from src.data import NFLTrajectoryDataset, collate_fn
    except ImportError as exc:  # pragma: no cover - dataset module may be missing during docs build.
        raise ImportError(
            "NFLTrajectoryDataset is required for training. Ensure src/data.py is implemented."
        ) from exc

    dataset_kwargs = dict(
        max_encoder_len=data_config.max_encoder_len,
        max_decoder_len=data_config.max_decoder_len,
        feature_cols=data_config.feature_cols,
        normalize=data_config.normalize,
        rotation_normalize=data_config.rotation_normalize,
    )

    train_dataset = NFLTrajectoryDataset(
        input_parquet=os.path.join(data_config.data_dir, "train_input.parquet"),
        output_parquet=os.path.join(data_config.data_dir, "train_output.parquet"),
        **dataset_kwargs,
    )
    val_dataset = NFLTrajectoryDataset(
        input_parquet=os.path.join(data_config.data_dir, "val_input.parquet"),
        output_parquet=os.path.join(data_config.data_dir, "val_output.parquet"),
        **dataset_kwargs,
    )

    transform_params = None
    get_params = getattr(train_dataset, "get_transform_params", None)
    if callable(get_params):
        transform_params = get_params()

    common_loader_kwargs = dict(
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=getattr(data_config, "pin_memory", False),
        collate_fn=collate_fn,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )
    return train_loader, val_loader, transform_params


def train_model(
    model_config: ModelConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
) -> Tuple[pl.Trainer, TrajectoryPredictionModule]:
    """Train the seq2seq model using PyTorch Lightning."""
    if training_config.seed is not None:
        pl.seed_everything(training_config.seed, workers=True)

    train_loader, val_loader, transform_params = _build_dataloaders(data_config)

    module = TrajectoryPredictionModule(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        transform_params=transform_params,
    )

    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=training_config.checkpoint_dir,
            monitor=training_config.monitor_metric,
            mode=training_config.monitor_mode,
            save_top_k=training_config.save_top_k,
            filename="approach1-{epoch:02d}-{val_rmse:.3f}",
        ),
        EarlyStopping(
            monitor=training_config.monitor_metric,
            mode=training_config.monitor_mode,
            patience=training_config.early_stopping_patience,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = _build_logger(training_config)
    if WandbLogger is not None and isinstance(logger, WandbLogger):
        logger.watch(module, log="all", log_freq=500)

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
    return trainer, module


__all__ = ["TrajectoryPredictionModule", "train_model"]
