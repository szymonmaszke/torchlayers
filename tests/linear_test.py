import torch

import torchlayers as tl


def test_last_dimension_linear():
    module = tl.Linear(64)
    module = tl.build(module, torch.randn(1, 3, 32, 32))
    assert module(torch.randn(2, 6, 24, 32)).shape == (2, 6, 24, 64)
