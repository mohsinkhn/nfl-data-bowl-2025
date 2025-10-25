"""PyTorch Lightning training utilities for the seq2seq trajectory baseline."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import math
import pytorch_lightning as pl
import torch
import numpy as np
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
from src.utils.normalization import rotate_coordinates, reconstruct_trajectory_from_polar
from src.model import (
    DirectTrajectoryModel,
    GRUDecoder,
    GRUEncoder,
    InputProjection,
    LSTMDecoder,
    LSTMEncoder,
    Seq2SeqTrajectoryModel,
    TargetReceiverDirectModel,
)


def _build_encoder(config: ModelConfig) -> nn.Module:
    """Instantiate the encoder specified by the model configuration."""
    if config.bidirectional:
        raise NotImplementedError(
            "Bidirectional encoders are not implemented for Approach 1"
        )

    input_dim = getattr(config, "encoder_input_dim", None) or (
        config.input_projection_dim if config.use_input_projection else config.input_dim
    )
    if config.encoder_type == "lstm":
        return LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    if config.encoder_type == "gru":
        return GRUEncoder(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported encoder type: {config.encoder_type}")


def _build_decoder(config: ModelConfig) -> nn.Module:
    """Instantiate the decoder specified by the model configuration."""
    # Decoder input_dim should be output_dim (x, y coordinates)
    if config.decoder_type == "lstm":
        return LSTMDecoder(
            input_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )
    if config.decoder_type == "gru":
        return GRUDecoder(
            input_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )
    raise ValueError(f"Unsupported decoder type: {config.decoder_type}")


def _load_pretrained_encoder(module: "TrajectoryPredictionModule", checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("state_dict", state)

    loaded_encoder = False
    for prefix in ("encoder.", "model.encoder."):
        encoder_state = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if encoder_state:
            current_state = module.encoder.state_dict()
            filtered_state = {}
            skipped = []
            for name, tensor in encoder_state.items():
                if name in current_state and current_state[name].shape == tensor.shape:
                    filtered_state[name] = tensor
                else:
                    skipped.append(name)
            missing, unexpected = module.encoder.load_state_dict(
                filtered_state, strict=False
            )
            if skipped:
                print(
                    "Warning: skipped encoder params due to shape mismatch: "
                    + ", ".join(skipped)
                )
            if missing:
                print(
                    f"Warning: missing encoder keys when loading pretrained weights: {missing}"
                )
            if unexpected:
                print(
                    f"Warning: unexpected encoder keys when loading pretrained weights: {unexpected}"
                )
            loaded_encoder = True
            break

    if not loaded_encoder:
        print(
            f"Warning: did not find encoder parameters in {checkpoint_path}; skipping load"
        )

    target_proj = getattr(module.model, "input_projection", None)
    if target_proj is not None:
        for prefix in ("input_projection.", "model.input_projection."):
            proj_state = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if proj_state:
                current_proj = target_proj.state_dict()
                filtered_proj = {}
                skipped_proj = []
                for name, tensor in proj_state.items():
                    if name in current_proj and current_proj[name].shape == tensor.shape:
                        filtered_proj[name] = tensor
                    else:
                        skipped_proj.append(name)
                missing, unexpected = target_proj.load_state_dict(
                    filtered_proj, strict=False
                )
                if skipped_proj:
                    print(
                        "Warning: skipped input_projection params due to shape mismatch: "
                        + ", ".join(skipped_proj)
                    )
                if missing:
                    print(
                        f"Warning: missing input_projection keys when loading pretrained weights: {missing}"
                    )
                if unexpected:
                    print(
                        f"Warning: unexpected input_projection keys when loading pretrained weights: {unexpected}"
                    )
                break


class EncoderNextFrameModule(pl.LightningModule):
    """Pretrain the encoder by predicting the next canonical frame."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> None:
        super().__init__()

        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)
        if isinstance(training_config, dict):
            training_config = TrainingConfig(**training_config)

        self.save_hyperparameters(
            {
                "model_config": asdict(model_config),
                "training_config": asdict(training_config),
            }
        )

        self.model_config = model_config
        self.training_config = training_config

        self.encoder = _build_encoder(model_config)
        self.input_projection = None
        if getattr(model_config, "use_input_projection", False):
            self.input_projection = InputProjection(
                input_dim=model_config.input_dim,
                proj_dim=model_config.input_projection_dim,
                use_batchnorm=model_config.input_projection_batchnorm,
                dropout=model_config.input_projection_dropout,
            )

        self.head = nn.Linear(model_config.hidden_dim, 2)

    def forward(
        self,
        encoder_inputs: Tensor,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.input_projection is not None:
            encoder_inputs = self.input_projection(encoder_inputs)
        outputs, _ = self.encoder(encoder_inputs, mask=encoder_mask)
        return self.head(outputs[:, :-1])

    def _masked_rmse(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        squared = (predictions - targets) ** 2
        if mask is not None:
            mask = mask.to(dtype=squared.dtype).unsqueeze(-1)
            numerator = (squared * mask).sum()
            denom = mask.sum() * predictions.size(-1)
        else:
            numerator = squared.sum()
            denom = torch.tensor(squared.numel(), device=squared.device, dtype=squared.dtype)
        denom = torch.clamp(denom, min=1.0)
        mse = numerator / denom
        return torch.sqrt(torch.clamp(mse, min=0.0))

    def _step(self, batch: Dict[str, Tensor], stage: str) -> Tensor:
        encoder_inputs = batch["encoder_input"]
        encoder_mask = batch.get("encoder_mask")

        predictions = self(encoder_inputs, encoder_mask)
        targets = encoder_inputs[:, 1:, :2]
        mask = encoder_mask[:, 1:] if encoder_mask is not None else None

        loss = self._masked_rmse(predictions, targets, mask)
        rmse = loss

        batch_size = encoder_inputs.size(0)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_rmse",
            rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
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

        if self.training_config.scheduler == "cosine_warmup":
            warmup_epochs = max(0, self.training_config.scheduler_warmup_epochs)
            total_epochs = max(1, self.training_config.max_epochs)

            def lr_lambda(epoch: int) -> float:
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return max(1e-8, (epoch + 1) / warmup_epochs)
                progress_numerator = max(0, epoch - warmup_epochs)
                progress_denominator = max(1, total_epochs - warmup_epochs)
                progress = min(progress_numerator / progress_denominator, 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return [optimizer], [scheduler]

        return optimizer


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

        # Handle both dataclass instances and dicts (for checkpoint loading)
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)
        if isinstance(training_config, dict):
            training_config = TrainingConfig(**training_config)
        if isinstance(data_config, dict):
            data_config = DataConfig(**data_config)

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
        self.prediction_mode = getattr(
            model_config, "prediction_mode", "autoregressive"
        )
        self.use_polar_targets = getattr(data_config, "use_polar_targets", False)

        encoder = _build_encoder(model_config)
        input_projection = None
        if getattr(model_config, "use_input_projection", False):
            input_projection = InputProjection(
                input_dim=model_config.input_dim,
                proj_dim=model_config.input_projection_dim,
                use_batchnorm=model_config.input_projection_batchnorm,
                dropout=model_config.input_projection_dropout,
            )

        if self.prediction_mode == "autoregressive":
            decoder = _build_decoder(model_config)
            self.model = Seq2SeqTrajectoryModel(encoder, decoder, input_projection)
            self.encoder = self.model.encoder
            self.decoder = self.model.decoder
        elif self.prediction_mode == "direct":
            projection_layers = getattr(model_config, "direct_projection_layers", 1)
            prediction_len = data_config.max_decoder_len
            self.model = DirectTrajectoryModel(
                encoder,
                hidden_dim=model_config.hidden_dim,
                output_dim=model_config.output_dim,
                prediction_len=prediction_len,
                projection_layers=projection_layers,
                input_projection=input_projection,
            )
            self.encoder = self.model.encoder
            self.decoder = None
        elif self.prediction_mode == "target_direct":
            prediction_len = data_config.max_decoder_len
            context_dim = getattr(model_config, "ball_context_dim", 0)
            context_start = getattr(model_config, "ball_context_start", -context_dim or -1)
            self.model = TargetReceiverDirectModel(
                encoder,
                hidden_dim=model_config.hidden_dim,
                output_dim=model_config.output_dim,
                prediction_len=prediction_len,
                context_dim=context_dim,
                context_start=context_start,
                dropout=model_config.dropout,
                input_projection=input_projection,
            )
            self.encoder = self.model.encoder
            self.decoder = None
        else:
            raise ValueError(f"Unsupported prediction_mode: {self.prediction_mode}")

    def forward(
        self,
        encoder_inputs: Tensor,
        decoder_inputs: Tensor,
        teacher_forcing_ratio: float,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.prediction_mode == "autoregressive":
            return self.model(
                encoder_inputs,
                decoder_inputs,
                teacher_forcing_ratio=teacher_forcing_ratio,
                encoder_mask=encoder_mask,
            )
        return self.model(encoder_inputs, encoder_mask=encoder_mask)

    def _masked_rmse(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        squared = (predictions - targets) ** 2
        if mask is not None:
            mask = mask.to(squared.dtype)
            expanded_mask = mask.unsqueeze(-1)
            squared = squared * expanded_mask
            valid_steps = expanded_mask.sum()
            normalizer = valid_steps * predictions.size(-1)
        else:
            normalizer = torch.tensor(
                squared.numel(), device=squared.device, dtype=squared.dtype
            )
        normalizer = torch.clamp(normalizer, min=1.0)
        mse = squared.sum() / normalizer
        return torch.sqrt(torch.clamp(mse, min=0.0))

    def _compute_physical_rmse(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Optional[Tensor],
        metadata: Optional[List[Dict[str, Any]]],
    ) -> Optional[Tensor]:
        """Compute RMSE in physical space using stored transforms."""
        if metadata is None:
            return None

        preds_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        if mask is not None:
            mask_np = mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.ones(preds_np.shape[:2], dtype=bool)

        total_sq = 0.0
        total_count = 0

        for i, meta in enumerate(metadata):
            if meta is None:
                continue
            mask_i = mask_np[i]
            if not mask_i.any():
                continue
            transform = meta.get("transform") if isinstance(meta, dict) else None

            if self.use_polar_targets:
                last_pos = np.array(meta["last_position_canonical"], dtype=np.float32)
                last_heading = float(meta["last_heading"])
                pred = reconstruct_trajectory_from_polar(
                    preds_np[i][mask_i], last_pos, last_heading
                )
                target = np.asarray(
                    meta["decoder_target_cartesian"], dtype=np.float32
                )[: pred.shape[0]]
            elif bool(meta.get("use_ball_residual_targets", False)):
                landing = np.array(meta["ball_land_canonical"], dtype=np.float32)
                pred = landing - preds_np[i][mask_i]
                target = landing - targets_np[i][mask_i]
            else:
                pred = preds_np[i][mask_i]
                target = targets_np[i][mask_i]

            if transform is not None:
                pred = transform.inverse_points(pred)
                target = transform.inverse_points(target)

            diff = pred - target
            total_sq += float(np.sum(diff**2))
            total_count += diff.shape[0]

        if total_count == 0:
            return None

        rmse = np.sqrt(total_sq / total_count / 2)  # Divide by 2 for x and y
        return torch.tensor(rmse, device=predictions.device, dtype=predictions.dtype)

    def training_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Tensor:  # noqa: D401
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
        loss = self._masked_rmse(predictions, decoder_targets, decoder_mask)
        rmse = loss

        physical_rmse = self._compute_physical_rmse(
            predictions,
            decoder_targets,
            decoder_mask,
            batch.get("metadata"),
        )

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
        if physical_rmse is not None:
            self.log(
                "train_rmse_physical",
                physical_rmse,
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
        loss = self._masked_rmse(predictions, decoder_targets, decoder_mask)
        rmse = loss

        physical_rmse = self._compute_physical_rmse(
            predictions,
            decoder_targets,
            decoder_mask,
            batch.get("metadata"),
        )

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
        if physical_rmse is not None:
            self.log(
                "val_rmse_physical",
                physical_rmse,
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

        steps = decoder_inputs.size(1)
        if self.prediction_mode == "autoregressive":
            if self.use_polar_targets:
                start_token = torch.zeros_like(decoder_inputs[:, 0])
            else:
                start_token = decoder_inputs[:, 0]
            predictions = self.model.predict(
                encoder_inputs,
                decoder_start=start_token,
                prediction_steps=steps,
                encoder_mask=encoder_mask,
            )
        else:
            predictions = self.model.predict(
                encoder_inputs,
                encoder_mask=encoder_mask,
            )
            if predictions.size(1) > steps:
                predictions = predictions[:, :steps]

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

        if self.training_config.scheduler == "cosine_warmup":
            warmup_epochs = max(0, self.training_config.scheduler_warmup_epochs)
            total_epochs = max(1, self.training_config.max_epochs)

            def lr_lambda(epoch: int) -> float:
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return max(1e-8, (epoch + 1) / warmup_epochs)
                progress_numerator = max(0, epoch - warmup_epochs)
                progress_denominator = max(1, total_epochs - warmup_epochs)
                progress = min(progress_numerator / progress_denominator, 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
        from src.data import (
            NFLTrajectoryDataset,
            TargetReceiverTrajectoryDataset,
            collate_fn,
        )
    except (
        ImportError
    ) as exc:  # pragma: no cover - dataset module may be missing during docs build.
        raise ImportError(
            "NFLTrajectoryDataset is required for training. Ensure src/data.py is implemented."
        ) from exc

    dataset_type = getattr(data_config, "dataset_type", "baseline")
    dataset_cls = (
        TargetReceiverTrajectoryDataset
        if dataset_type == "target_receiver"
        else NFLTrajectoryDataset
    )

    dataset_kwargs = dict(
        max_encoder_len=data_config.max_encoder_len,
        max_decoder_len=data_config.max_decoder_len,
        feature_cols=data_config.feature_cols,
        normalize=data_config.normalize,
        rotation_normalize=data_config.rotation_normalize,
        align_heading=getattr(data_config, "align_heading", True),
        use_player_role=getattr(data_config, "use_player_role", True),
        use_player_attributes=getattr(data_config, "use_player_attributes", True),
        use_polar_targets=getattr(data_config, "use_polar_targets", False),
        use_ball_residual_targets=getattr(
            data_config, "use_ball_residual_targets", False
        ),
        vertical_flip_prob=getattr(
            data_config, "vertical_flip_prob", 0.0
        ),
    )
    aug_deg = getattr(data_config, "rotation_augmentation_degrees", 45.0)

    target_specific_kwargs = {}
    if dataset_cls is TargetReceiverTrajectoryDataset:
        dataset_kwargs.pop("use_player_role", None)
        target_specific_kwargs.update(
            dict(
                min_decoder_len=getattr(data_config, "target_min_decoder_len", 1),
                filter_target_only=getattr(data_config, "target_only", True),
            )
        )

    train_dataset = dataset_cls(
        input_parquet=os.path.join(data_config.data_dir, "train_input.parquet"),
        output_parquet=os.path.join(data_config.data_dir, "train_output.parquet"),
        rotation_augmentation_prob=getattr(
            data_config, "rotation_augmentation_prob", 0.0
        ),
        rotation_augmentation_degrees=aug_deg,
        **dataset_kwargs,
        **target_specific_kwargs,
    )

    val_kwargs = dict(dataset_kwargs)
    val_kwargs["rotation_augmentation_prob"] = 0.0
    val_kwargs["vertical_flip_prob"] = 0.0

    val_dataset = dataset_cls(
        input_parquet=os.path.join(data_config.data_dir, "val_input.parquet"),
        output_parquet=os.path.join(data_config.data_dir, "val_output.parquet"),
        **val_kwargs,
        **target_specific_kwargs,
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
    pretrained_encoder_path: Optional[str] = None,
) -> Tuple[pl.Trainer, TrajectoryPredictionModule]:
    """Train the seq2seq model using PyTorch Lightning."""
    if training_config.seed is not None:
        pl.seed_everything(training_config.seed, workers=True)

    train_loader, val_loader, transform_params = _build_dataloaders(data_config)

    train_dataset = train_loader.dataset
    if hasattr(train_dataset, "input_dim"):
        model_config.input_dim = train_dataset.input_dim
    if hasattr(train_dataset, "output_dim"):
        model_config.output_dim = train_dataset.output_dim

    module = TrajectoryPredictionModule(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        transform_params=transform_params,
    )

    if pretrained_encoder_path is not None:
        print(f"Loading pretrained encoder weights from {pretrained_encoder_path}")
        _load_pretrained_encoder(module, pretrained_encoder_path)

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


__all__ = ["EncoderNextFrameModule", "TrajectoryPredictionModule", "train_model"]
