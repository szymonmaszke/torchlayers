import torch

import pytest
import torchlayers


class _CustomLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.some_params = torch.nn.Parameter(torch.randn(2, out_features))


CustomLinear = torchlayers.make_inferrable(_CustomLinear)


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


def test_custom_inferable():
    layer = CustomLinear(32)
    assert layer(torch.rand(16, 64)).shape == (16, 64)


def test_custom_inferable():
    layer = CustomLinear(32)
    layer(torch.rand(16, 64))
    assert layer.some_params.shape == (2, 32)
