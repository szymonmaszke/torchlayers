import torch

import pytest
import torchlayers as tl


@pytest.mark.parametrize("klass", ("Swish", "HardSwish", "HardSigmoid"))
def test_object(klass):
    getattr(tl, klass)()(torch.randn(4, 5, 6))


@pytest.mark.parametrize("function", ("swish", "hard_swish", "hard_sigmoid"))
def test_functional(function):
    getattr(tl, function)(torch.randn(4, 5, 6))
