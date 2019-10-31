import pathlib
import tempfile

import torch

import torchlayers


def test_basic():
    pass
    # inputs = torch.randn(16, 3, 32, 32)
    # temp = pathlib.Path(tempfile.gettempdir())

    # layer = torchlayers.Conv(64)
    # output = layer(inputs)
    # scripted = torch.jit.script(layer)
    # torch.save(layer, temp / "conv_model.pt")

    # new_layer = torch.load(temp / "conv_model.pt")
    # new_output = new_layer(inputs)
    # assert torch.allclose(output, new_output)
