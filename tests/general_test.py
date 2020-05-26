import torch

import pytest
import torchlayers as tl


class ConcatenateProxy(torch.nn.Module):
    # Return same tensor three times
    # You could explicitly return a list or tuple as well
    def forward(self, tensor):
        return tensor, tensor, tensor


class _CustomLinearImpl(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.some_params = torch.nn.Parameter(torch.randn(2, out_features))


CustomLinear = tl.infer(_CustomLinearImpl)


@pytest.fixture
def model():
    return tl.Sequential(
        tl.Conv(64),
        tl.ReLU(),
        tl.MaxPool(),
        tl.BatchNorm(),
        tl.Conv(128),
        tl.ReLU(),
        tl.MaxPool(),
        tl.Conv(256),
        tl.ReLU(),
        tl.Reshape(-1),
    )


def test_reshape(model):
    assert model(torch.randn(16, 1, 32, 32)).shape == (16, 256 * 8 * 8)


def test_lambda():
    layer = tl.Lambda(lambda inputs: inputs * 3)
    output = layer(torch.ones(16))
    assert torch.sum(output) == 16 * 3


def test_concatenate():
    model = torch.nn.Sequential(ConcatenateProxy(), tl.Concatenate(dim=-1))
    assert model(torch.randn(64, 20)).shape == torch.randn(64, 60).shape


def test_custom_inferable_output():
    layer = CustomLinear(out_features=32)
    assert layer(torch.rand(16, 64)).shape == torch.randn(16, 32).shape


def test_custom_inferable_parameters():
    layer = CustomLinear(32)
    layer(torch.rand(16, 64))
    assert layer.some_params.shape == (2, 32)


def test_custom_inferable_build():
    layer = CustomLinear(32)
    layer = tl.build(layer, torch.rand(16, 64))
    assert layer.some_params.shape == (2, 32)


def test_smoke_summary(model):
    summary = tl.summary(model, torch.randn(1, 3, 32, 32)).string(
        inputs=False, buffers=False
    )
    assert (len(summary)) == 1459
