import torch

import torchlayers as tl


def test_module():
    layer = tl.Conv2d(64, kernel_size=3)
    assert layer.__module__ == "torchlayers"
