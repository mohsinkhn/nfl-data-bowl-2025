"""Seq2Seq encoder-decoder models for NFL trajectory prediction."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _mask_to_lengths(mask: Optional[Tensor], seq_len: int) -> Optional[Tensor]:
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        mask = mask != 0
    lengths = mask.sum(dim=1)
    return lengths.clamp(min=1).cpu()


class BaseRecurrentEncoder(nn.Module):
    def __init__(self, rnn: nn.Module):
        super().__init__()
        self.rnn = rnn

    def forward(
        self, inputs: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        lengths = _mask_to_lengths(mask, inputs.size(1))
        if lengths is not None:
            packed = pack_padded_sequence(
                inputs, lengths, batch_first=True, enforce_sorted=False
            )
            outputs, hidden = self.rnn(packed)
            outputs, _ = pad_packed_sequence(
                outputs, batch_first=True, total_length=inputs.size(1)
            )
        else:
            outputs, hidden = self.rnn(inputs)
        return outputs, hidden


class LSTMEncoder(BaseRecurrentEncoder):
    """LSTM encoder returning sequence outputs and (h_n, c_n)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ):
        lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        super().__init__(lstm)


class GRUEncoder(BaseRecurrentEncoder):
    """GRU encoder returning sequence outputs and h_n."""

    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float
    ):
        gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        super().__init__(gru)


class BaseDecoder(nn.Module):
    def __init__(
        self,
        rnn: nn.Module,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.rnn = rnn
        self.output_head = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        if input_dim == output_dim:
            self.feedback = nn.Identity()
        else:
            self.feedback = nn.Linear(output_dim, input_dim)

    def forward(
        self,
        decoder_inputs: Tensor,
        hidden: Tuple[Tensor, ...] | Tensor,
        teacher_forcing_ratio: float = 0.0,
    ) -> Tuple[Tensor, Tuple[Tensor, ...] | Tensor]:
        batch, seq_len, _ = decoder_inputs.size()
        ratio = float(max(0.0, min(1.0, teacher_forcing_ratio)))
        rnn_input = decoder_inputs[:, 0]
        outputs = []
        state = hidden
        for t in range(seq_len):
            step_input = rnn_input.unsqueeze(1)
            step_output, state = self.rnn(step_input, state)
            logits = self.output_head(self.dropout(step_output.squeeze(1)))
            outputs.append(logits)
            if t + 1 < seq_len:
                if ratio > 0.0:
                    teacher_mask = (
                        torch.rand(batch, device=decoder_inputs.device) < ratio
                    )
                    teacher_next = decoder_inputs[:, t + 1]
                    model_next = self.feedback(logits)
                    rnn_input = torch.where(
                        teacher_mask.unsqueeze(-1), teacher_next, model_next
                    )
                else:
                    rnn_input = self.feedback(logits)
        stacked = torch.stack(outputs, dim=1)
        return stacked, state

    def generate(
        self,
        start_input: Tensor,
        hidden: Tuple[Tensor, ...] | Tensor,
        steps: int,
    ) -> Tensor:
        outputs = []
        rnn_input = start_input
        state = hidden
        for _ in range(steps):
            step_output, state = self.rnn(rnn_input.unsqueeze(1), state)
            logits = self.output_head(step_output.squeeze(1))
            outputs.append(logits)
            rnn_input = self.feedback(logits)
        return torch.stack(outputs, dim=1)


class LSTMDecoder(BaseDecoder):
    """LSTM decoder with optional teacher forcing support."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
    ):
        lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        super().__init__(lstm, input_dim, hidden_dim, output_dim, dropout)


class GRUDecoder(BaseDecoder):
    """GRU decoder with optional teacher forcing support."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
    ):
        gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        super().__init__(gru, input_dim, hidden_dim, output_dim, dropout)


class InputProjection(nn.Module):
    """Project heterogeneous input features before the temporal encoder."""

    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, proj_dim)
        self.batchnorm = nn.BatchNorm1d(proj_dim) if use_batchnorm else None
        self.activation = activation if activation is not None else nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        batch, seq_len, feat = inputs.shape
        x = inputs.view(batch * seq_len, feat)
        x = self.linear(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x.view(batch, seq_len, -1)


class Seq2SeqTrajectoryModel(nn.Module):
    """Encodes pre-throw sequences and decodes future trajectories."""

    def __init__(
        self,
        encoder: BaseRecurrentEncoder,
        decoder: BaseDecoder,
        input_projection: Optional[InputProjection] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_projection = input_projection

    def forward(
        self,
        encoder_inputs: Tensor,
        decoder_inputs: Tensor,
        teacher_forcing_ratio: float = 0.0,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.input_projection is not None:
            encoder_inputs = self.input_projection(encoder_inputs)
        _, hidden = self.encoder(encoder_inputs, mask=encoder_mask)
        outputs, _ = self.decoder(decoder_inputs, hidden, teacher_forcing_ratio)
        return outputs

    def predict(
        self,
        encoder_inputs: Tensor,
        decoder_start: Tensor,
        prediction_steps: int,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        self.eval()
        with torch.no_grad():
            if self.input_projection is not None:
                encoder_inputs = self.input_projection(encoder_inputs)
            _, hidden = self.encoder(encoder_inputs, mask=encoder_mask)
            predictions = self.decoder.generate(decoder_start, hidden, prediction_steps)
        return predictions


class DirectTrajectoryModel(nn.Module):
    """Predicts future trajectories without autoregressive decoding."""

    def __init__(
        self,
        encoder: BaseRecurrentEncoder,
        hidden_dim: int,
        output_dim: int,
        prediction_len: int,
        projection_layers: int = 1,
        input_projection: Optional[InputProjection] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim
        self.prediction_len = prediction_len
        self.input_projection = input_projection

        layers = []
        in_dim = hidden_dim
        for _ in range(max(0, projection_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, prediction_len * output_dim))
        self.projection = nn.Sequential(*layers)

    def forward(
        self,
        encoder_inputs: Tensor,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.input_projection is not None:
            encoder_inputs = self.input_projection(encoder_inputs)
        _, hidden = self.encoder(encoder_inputs, mask=encoder_mask)
        if isinstance(hidden, tuple):
            hidden_state = hidden[0]
        else:
            hidden_state = hidden
        # last_hidden = hidden_state[-1]  # (batch, hidden_dim)
        # use average of all hidden states
        last_hidden = hidden_state[-5:].mean(dim=0)
        flat = self.projection(last_hidden)
        return flat.view(-1, self.prediction_len, self.output_dim)

    def predict(
        self,
        encoder_inputs: Tensor,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(encoder_inputs, encoder_mask=encoder_mask)


class TargetReceiverDirectModel(nn.Module):
    """Direct regressor tailored for target receivers using pooled encoder context."""

    def __init__(
        self,
        encoder: BaseRecurrentEncoder,
        hidden_dim: int,
        output_dim: int,
        prediction_len: int,
        context_dim: int = 0,
        context_start: int = -4,
        dropout: float = 0.1,
        input_projection: Optional[InputProjection] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim
        self.prediction_len = prediction_len
        self.input_projection = input_projection
        self.context_dim = max(0, int(context_dim))
        self.context_slice = self._resolve_context_slice(context_start, self.context_dim)

        fusion_in = hidden_dim + (self.context_dim if self.context_slice is not None else 0)
        self.fusion = nn.Linear(fusion_in, hidden_dim)
        self.activation = nn.GELU()
        self.pre_dropout = nn.Dropout(dropout if dropout > 0.0 else 0.0)
        self.post_dropout = nn.Dropout(dropout if dropout > 0.0 else 0.0)
        self.output_head = nn.Linear(hidden_dim, prediction_len * output_dim)

    @staticmethod
    def _resolve_context_slice(start: int, dim: int) -> Optional[slice]:
        if dim <= 0:
            return None
        if start >= 0:
            return slice(start, start + dim)
        return slice(start, None)

    def _pool_encoder_outputs(
        self, encoder_outputs: Tensor, encoder_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        if encoder_mask is not None:
            mask = encoder_mask.unsqueeze(-1).to(encoder_outputs.dtype)
            totals = (encoder_outputs * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = totals / counts
            last_indices = encoder_mask.long().sum(dim=1) - 1
            last_indices = last_indices.clamp(min=0)
        else:
            pooled = encoder_outputs.mean(dim=1)
            seq_len = encoder_outputs.size(1)
            last_indices = torch.full(
                (encoder_outputs.size(0),),
                max(seq_len - 1, 0),
                device=encoder_outputs.device,
                dtype=torch.long,
            )
        return pooled, last_indices

    def forward(
        self,
        encoder_inputs: Tensor,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        raw_inputs = encoder_inputs
        if self.input_projection is not None:
            encoder_inputs = self.input_projection(encoder_inputs)

        encoder_outputs, _ = self.encoder(encoder_inputs, mask=encoder_mask)
        pooled, last_indices = self._pool_encoder_outputs(encoder_outputs, encoder_mask)

        if self.context_slice is not None:
            batch_indices = torch.arange(
                raw_inputs.size(0), device=raw_inputs.device, dtype=torch.long
            )
            context = raw_inputs[batch_indices, last_indices, self.context_slice]
            if context.ndim == 1:
                context = context.unsqueeze(-1)
            fused = torch.cat([pooled, context], dim=-1)
        else:
            fused = pooled

        fused = self.pre_dropout(fused)
        fused = self.activation(self.fusion(fused))
        fused = self.post_dropout(fused)
        flat = self.output_head(fused)
        return flat.view(-1, self.prediction_len, self.output_dim)

    def predict(
        self,
        encoder_inputs: Tensor,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(encoder_inputs, encoder_mask=encoder_mask)


__all__ = [
    "LSTMEncoder",
    "GRUEncoder",
    "LSTMDecoder",
    "GRUDecoder",
    "Seq2SeqTrajectoryModel",
    "InputProjection",
    "DirectTrajectoryModel",
    "TargetReceiverDirectModel",
]
