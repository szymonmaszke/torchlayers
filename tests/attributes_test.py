import torch

import torchlayers


def test_module():
    layer = torchlayers.Conv2d(64, kernel_size=3)
    assert layer.__module__ == "torchlayers"
