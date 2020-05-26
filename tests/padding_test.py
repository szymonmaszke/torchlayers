import torch

import torchlayers as tl


def test_odd_padding():
    for kernel_size in range(1, 14, 2):
        layer = tl.Conv(64, kernel_size=kernel_size)
        assert layer(torch.rand(1, 3, 28, 28)).shape == (1, 64, 28, 28)
