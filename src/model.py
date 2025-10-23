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

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
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

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
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
                    teacher_mask = torch.rand(
                        batch, device=decoder_inputs.device
                    ) < ratio
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


class Seq2SeqTrajectoryModel(nn.Module):
    """Encodes pre-throw sequences and decodes future trajectories."""

    def __init__(
        self,
        encoder: BaseRecurrentEncoder,
        decoder: BaseDecoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        encoder_inputs: Tensor,
        decoder_inputs: Tensor,
        teacher_forcing_ratio: float = 0.0,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
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
            _, hidden = self.encoder(encoder_inputs, mask=encoder_mask)
            predictions = self.decoder.generate(decoder_start, hidden, prediction_steps)
        return predictions


__all__ = [
    "LSTMEncoder",
    "GRUEncoder",
    "LSTMDecoder",
    "GRUDecoder",
    "Seq2SeqTrajectoryModel",
]
