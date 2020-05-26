import pathlib
import tempfile

import torch

import torchlayers as tl


def test_basic_jit():
    inputs = torch.randn(16, 3, 32, 32)

    layer = tl.build(tl.Conv(64), inputs)
    output = layer(inputs)
    new_model = torch.jit.script(layer)

    new_output = new_model(inputs)
    assert torch.allclose(output, new_output)


def test_basic_jit_save():
    inputs = torch.randn(16, 3, 32, 32)
    temp = pathlib.Path(tempfile.gettempdir())

    layer = tl.build(tl.Conv(64), inputs)
    output = layer(inputs)
    new_model = torch.jit.script(layer)

    torch.jit.save(new_model, str(temp / "jit.pt"))

    loaded_model = torch.jit.load(str(temp / "jit.pt"))
    new_output = loaded_model(inputs)
    assert torch.allclose(output, new_output)


# Test with JIT all custom modules
