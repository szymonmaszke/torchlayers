import torch

import pytest
import torchlayers


def test_gru():
    model = torchlayers.build(
        torchlayers.GRU(hidden_size=40, dropout=0.5, batch_first=True, num_layers=3),
        torch.randn(5, 3, 10),
    )

    output, _ = model(torch.randn(5, 3, 10))
    assert output.shape == (5, 3, 40)


def test_lstm():
    model = torchlayers.build(torchlayers.LSTM(hidden_size=20), torch.randn(5, 3, 10))

    output, (_, _) = model(torch.randn(5, 3, 10))
    assert output.shape == (5, 3, 20)
