import torch

import pytest
import torchlayers


class ConcatenateProxy(torch.nn.Module):
    # Return same tensor three times
    # You could explicitly return a list or tuple as well
    def forward(self, tensor):
        return tensor, tensor, tensor


@torchlayers.Infer()
class CustomLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.some_params = torch.nn.Parameter(torch.randn(2, out_features))


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
        torchlayers.Reshape(-1),
    )


def test_reshape(model):
    assert model(torch.randn(16, 1, 32, 32)).shape == (16, 256 * 8 * 8)


def test_lambda():
    layer = torchlayers.Lambda(lambda inputs: inputs * 3)
    output = layer(torch.ones(16))
    assert torch.sum(output) == 16 * 3


def test_concatenate():
    model = torch.nn.Sequential(ConcatenateProxy(), torchlayers.Concatenate(dim=-1))
    assert model(torch.randn(64, 20)).shape == torch.randn(64, 60).shape


def test_custom_inferable_output():
    layer = CustomLinear(out_features=32)
    assert layer(torch.rand(16, 64)).shape == torch.randn(16, 32).shape


def test_custom_inferable_parameters():
    layer = CustomLinear(32)
    layer(torch.rand(16, 64))
    assert layer.some_params.shape == (2, 32)
