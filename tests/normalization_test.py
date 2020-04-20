import torch

import pytest
import torchlayers


def test_2dbatchnorm():
    layer = torchlayers.BatchNorm()
    layer(torch.randn(20, 100))
