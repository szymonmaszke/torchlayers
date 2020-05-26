import torch

import pytest
import torchlayers as tl


def test_2dbatchnorm():
    layer = tl.BatchNorm()
    layer(torch.randn(20, 100))
