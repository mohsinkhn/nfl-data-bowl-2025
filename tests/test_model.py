import torch

from src.model import (
    DirectTrajectoryModel,
    GRUDecoder,
    GRUEncoder,
    LSTMDecoder,
    LSTMEncoder,
    Seq2SeqTrajectoryModel,
)


def _dummy_inputs(batch_size=2, enc_len=5, dec_len=4, input_dim=6, output_dim=2):
    encoder_inputs = torch.randn(batch_size, enc_len, input_dim)
    decoder_inputs = torch.randn(batch_size, dec_len, output_dim)
    encoder_mask = torch.ones(batch_size, enc_len, dtype=torch.bool)
    return encoder_inputs, decoder_inputs, encoder_mask


def test_seq2seq_forward_shapes():
    encoder_inputs, decoder_inputs, encoder_mask = _dummy_inputs()
    encoder = LSTMEncoder(input_dim=6, hidden_dim=16, num_layers=2, dropout=0.1)
    decoder = LSTMDecoder(
        input_dim=2,
        hidden_dim=16,
        num_layers=2,
        dropout=0.1,
        output_dim=2,
    )
    model = Seq2SeqTrajectoryModel(encoder, decoder)

    outputs = model(
        encoder_inputs=encoder_inputs,
        decoder_inputs=decoder_inputs,
        teacher_forcing_ratio=0.5,
        encoder_mask=encoder_mask,
    )
    assert outputs.shape == decoder_inputs.shape


def test_seq2seq_predict_autoregressive():
    encoder_inputs, decoder_inputs, encoder_mask = _dummy_inputs()
    encoder = GRUEncoder(input_dim=6, hidden_dim=8, num_layers=1, dropout=0.0)
    decoder = GRUDecoder(
        input_dim=2,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        output_dim=2,
    )
    model = Seq2SeqTrajectoryModel(encoder, decoder)

    start_token = decoder_inputs[:, 0]
    prediction_steps = decoder_inputs.size(1)
    outputs = model.predict(
        encoder_inputs=encoder_inputs,
        decoder_start=start_token,
        prediction_steps=prediction_steps,
        encoder_mask=encoder_mask,
    )

    assert outputs.shape == torch.Size([encoder_inputs.size(0), prediction_steps, 2])


def test_teacher_forcing_ratio_bounds():
    encoder_inputs, decoder_inputs, encoder_mask = _dummy_inputs()
    encoder = LSTMEncoder(input_dim=6, hidden_dim=8, num_layers=1, dropout=0.0)
    decoder = LSTMDecoder(
        input_dim=2,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        output_dim=2,
    )
    model = Seq2SeqTrajectoryModel(encoder, decoder)

    # ratio should be clamped to [0, 1]
    outputs = model(
        encoder_inputs=encoder_inputs,
        decoder_inputs=decoder_inputs,
        teacher_forcing_ratio=5.0,
        encoder_mask=encoder_mask,
    )
    assert outputs.shape == decoder_inputs.shape


def test_direct_trajectory_model_shapes():
    encoder_inputs, decoder_inputs, encoder_mask = _dummy_inputs()
    prediction_len = decoder_inputs.size(1)
    encoder = GRUEncoder(input_dim=6, hidden_dim=16, num_layers=2, dropout=0.1)
    model = DirectTrajectoryModel(
        encoder=encoder,
        hidden_dim=16,
        output_dim=2,
        prediction_len=prediction_len,
    )

    outputs = model(encoder_inputs, encoder_mask=encoder_mask)
    assert outputs.shape == torch.Size([encoder_inputs.size(0), prediction_len, 2])

    predictions = model.predict(encoder_inputs, encoder_mask=encoder_mask)
    assert predictions.shape == torch.Size([encoder_inputs.size(0), prediction_len, 2])
