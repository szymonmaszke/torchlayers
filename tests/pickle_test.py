import pathlib
import tempfile

import torch

import torchlayers


def test_save():
    inputs = torch.randn(16, 32)
    temp = pathlib.Path(tempfile.gettempdir())

    layer = torchlayers.build(torchlayers.Linear(64), inputs)
    output = layer(inputs)
    torch.save(layer, temp / "linear_model.pt")

    new_layer = torch.load(temp / "linear_model.pt")
    new_output = new_layer(inputs)
    assert torch.allclose(output, new_output)


def test_convolution_save():
    inputs = torch.randn(16, 3, 32, 32)
    temp = pathlib.Path(tempfile.gettempdir())

    layer = torchlayers.build(torchlayers.Conv2d(64, kernel_size=3), inputs)
    output = layer(inputs)
    torch.save(layer, temp / "conv_model.pt")

    new_layer = torch.load(temp / "conv_model.pt")
    new_output = new_layer(inputs)
    assert torch.allclose(output, new_output)


def test_dimension_save():
    inputs = torch.randn(16, 3, 32, 32)
    temp = pathlib.Path(tempfile.gettempdir())

    layer = torchlayers.build(torchlayers.Conv(64), inputs)
    output = layer(inputs)
    torch.save(layer, temp / "conv_model.pt")

    new_layer = torch.load(temp / "conv_model.pt")
    new_output = new_layer(inputs)
    assert torch.allclose(output, new_output)
