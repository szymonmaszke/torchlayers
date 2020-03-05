import torch

import pytest
import torchlayers


def test_invalid_dimension_batchnorm():
    with pytest.raises(ValueError):
        layer = torchlayers.BatchNorm()
        layer(torch.randn(1, 2, 3, 4, 5, 6, 7, 8))


def test_2dbatchnorm():
    layer = torchlayers.BatchNorm()
    layer(torch.randn(20, 100))
