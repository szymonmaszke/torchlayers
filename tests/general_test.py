import torch

import pytest
import torchlayers


@pytest.fixture
def model():
    return torchlayers.Sequential(
        torchlayers.Conv(64),
        torchlayers.ReLU(),
        torchlayers.MaxPool(),
        torchlayers.Conv(128),
        torchlayers.ReLU(),
        torchlayers.MaxPool(),
        torchlayers.Conv(256),
        torchlayers.ReLU(),
        torchlayers.Flatten(),
    )


def test_flatten(model):
    assert model(torch.randn(16, 1, 32, 32)).shape == (16, 256 * 8 * 8)


def test_lambda():
    layer = torchlayers.Lambda(lambda inputs: inputs * 3)
    output = layer(torch.ones(16))
    assert torch.sum(output) == 16 * 3
